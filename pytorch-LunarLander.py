import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import random
import os

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    
setup_seed(10)

def simulate_with_trained_model(agent, max_step=1000, save_name=None):
    sim_env = gym.make('LunarLander-v2', render_mode='rgb_array') 
    if(save_name == None):
        save_name = agent.n_game
    sim_env = gym.wrappers.RecordVideo(sim_env, video_folder=f"./{save_name}", name_prefix="after_training",episode_trigger=lambda x: True)
    state, _ = sim_env.reset()
    done = False
    step = 0
    while not done and step < max_step:
        action = agent.get_action(state, float('inf')) 
        state, _, done, _, _ = sim_env.step(action)
        step += 1
    sim_env.close()
    
class PrioritizedReplayBuffer:
    def __init__(self, 
                 max_samples=10000,
                 state_dim=8,
                 alpha=0.6,
                 beta0=0.1,
                 beta_rate=0.999,
                 eps=1e-6,
                 device='cuda'):
        self.device = device
        self.max_samples = max_samples
        self.state_dim = state_dim
        self.states = torch.empty((max_samples, state_dim), dtype=torch.float32, device=device)
        self.next_states = torch.empty((max_samples, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.empty(max_samples, dtype=torch.long, device=device)  # Assuming actions are discrete
        self.rewards = torch.empty(max_samples, dtype=torch.float32, device=device)
        self.dones = torch.empty(max_samples, dtype=torch.bool, device=device)
        self.priorities = torch.zeros(max_samples, dtype=torch.float32, device=device)
        self.n_entries = 0
        self.next_index = 0
        self.alpha = alpha
        self.beta = beta0
        self.beta0 = beta0
        self.beta_rate = beta_rate
        self.eps = eps

    def update(self, indices, priorities):
        self.priorities[indices] = priorities

    def store(self, sample):
        state, action, reward, next_state, done = sample
        self.states[self.next_index] = torch.tensor(state, device=self.device)
        self.actions[self.next_index] = torch.tensor(action, device=self.device)
        self.rewards[self.next_index] = torch.tensor(reward, device=self.device)
        self.next_states[self.next_index] = torch.tensor(next_state, device=self.device)
        self.dones[self.next_index] = torch.tensor(done, device=self.device)

        priority = 1.0 if self.n_entries == 0 else self.priorities[:self.n_entries].max().item()
        self.priorities[self.next_index] = priority
        self.n_entries = min(self.n_entries + 1, self.max_samples)
        self.next_index = (self.next_index + 1) % self.max_samples

    def sample(self, batch_size):
        # self.beta = min(1.0, self.beta / self.beta_rate)
        # priorities = self.priorities[:self.n_entries] if self.n_entries < self.max_samples else self.priorities
        # scaled_priorities = priorities.pow(self.alpha)
        # probs = scaled_priorities / scaled_priorities.sum()
        # indices = torch.multinomial(probs, batch_size, replacement=False).to(self.device)
        indices = torch.randint(0, self.n_entries, (batch_size,), dtype=torch.long, device=self.device)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        # weights = (1 / (self.n_entries * probs[indices])).pow(self.beta)
        # normalized_weights = weights / weights.max()

        # return indices, normalized_weights, (states, actions, rewards, next_states, dones)
        return indices, (states, actions, rewards, next_states, dones)
    
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.linear4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hidden = F.relu(self.linear1(x))
        hidden = F.relu(self.linear2(hidden))
        advantage = self.linear3(hidden)
        value = self.linear4(hidden)
        value = value.expand_as(advantage)
        qvalue = value + advantage - advantage.mean(-1, keepdim=True).expand_as(advantage)
        return qvalue
    
class QTrainer:
    def __init__(self, lr, gamma,input_dim, hidden_dim, output_dim, device, tau=0.005):
        self.tau = tau 
        self.device = device
        self.gamma = gamma
        self.model = Linear_QNet(input_dim,hidden_dim,output_dim)
        self.target_model = Linear_QNet(input_dim,hidden_dim,output_dim)
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.copy_model()
    
    def soft_update(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def copy_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        Q_value = self.model(state).gather(-1, action.unsqueeze(-1)).squeeze()
        Q_value_next_index = self.model(next_state).detach().max(-1)[1]
        Q_value_next_target = self.target_model(next_state).detach().squeeze()
        Q_value_next = Q_value_next_target.gather(-1, Q_value_next_index.unsqueeze(-1)).squeeze()

        target =  (reward + self.gamma * Q_value_next * ~done)

        self.optimizer.zero_grad()
        loss = self.criterion(target, Q_value)
        loss.backward()
        self.optimizer.step()

        return torch.abs(target - Q_value).detach()

class Agent:
    def __init__(self,state_space, state_dim, action_space, hidden_dim = 64, max_explore=1000, gamma = 0.995,
                max_memory=100_000, lr=1e-3, device = 'cuda'):
        self.device = device
        self.max_explore = max_explore 
        self.PRB = PrioritizedReplayBuffer(max_samples=max_memory,state_dim=state_dim, device=self.device)
        self.nS = state_space  
        self.nA = action_space  
        self.step = 0
        self.n_game = 0
        self.trainer = QTrainer(lr, gamma, self.nS, hidden_dim, self.nA, device=self.device)

    def remember(self, state, action, reward, next_state, done):
        self.PRB.store((state, action, reward, next_state, done))

    def train_long_memory(self,batch_size):
        if self.PRB.n_entries > batch_size:
            idxs, experiences = self.PRB.sample(batch_size)
            states, actions, rewards, next_states, dones = experiences
            priorities = self.trainer.train_step(states, actions, rewards, next_states, dones)
            self.PRB.update(idxs, priorities)

    def get_action(self, state, n_game):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        prediction = self.trainer.model(state)
        
        epsilon = max(self.max_explore - n_game, 10)
        if random.randint(0, self.max_explore) < epsilon:
            final_move = np.random.randint(self.nA)
        else:
            final_move = prediction.argmax().item()
        return final_move

def train(env, max_game=2000, max_step=1000, device='cuda'):
    agent = Agent(state_space = env.observation_space.shape[0], 
                  state_dim=8,
                action_space = env.action_space.n,
                hidden_dim=64,
                max_explore=1000, gamma = 0.995,
                max_memory=100_000, lr=1e-3, device=device)
    results = []
    state_new, _ = env.reset()
    done = False
    total_step = 0
    total_points = 0
    while agent.n_game <= max_game:
        state_old = state_new.copy()
        action = agent.get_action(state_old, agent.n_game)
        state_new, reward, done, _, _ = env.step(action)
        agent.remember(state_old, action, reward, state_new, done)

        agent.step += 1
        total_step += 1

        if total_step % 4 == 0:
            agent.train_long_memory(batch_size=64)
            agent.trainer.soft_update()
        total_points += reward
        if done or agent.step>max_step:
            results.append(total_points)
            total_points = 0
            mean_result = np.mean(results[-100:])
            state_new, _ = env.reset()
            agent.step = 0
            agent.n_game += 1
            if (agent.n_game>0) and (agent.n_game % 200 ==0):         
                print("Running episode  {}, step {} Mean goal {:.2f}. ".format(
                    agent.n_game, total_step, mean_result))
                simulate_with_trained_model(agent, max_step) 
            if mean_result >= 200.0:
                break
    env.close()
    return agent


env = gym.make('LunarLander-v2', render_mode='rgb_array')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
agent = train(env, 2000, device=device)
simulate_with_trained_model(agent, save_name='finish') 