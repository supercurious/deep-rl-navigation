import numpy as np
import random
from collections import namedtuple, deque
from SumTree import SumTree
from model import QNet, QNet_Duel
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Agent interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, 
                 buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3, 
                 learn_rate=5e-4, update_every=4, replace_every=32,
                 doubleDQN=True, soft=True, duel=False, prioritized=False,
                 alpha=0, beta_start=0, beta_end=0, beta_episodes=300,
                 hidden_layers=[128,64]):
        """
        Initialize an Agent object.
        
        # Parameters
            state_size (int): Size of state dimension
            action_size (int): Size of action dimension
            seed (int): Random seed
            buffer_size (int): Replay buffer size
            batch_size (int): Minibatch size
            gamma (float): Discount factor
            tau (float): For soft update of target parameters
            LR (float): Learning rate
            update_every (int): How often to update the network
            replace_every (int): How often to replace network (hard update)
            algo (str): Algorithm name ("DQN", "DoubleDQN")
            soft (bool): True for soft updates, false for periodic replace.
            duel (bool): True for dueling network architecture
            prioritized (bool): True to enable prioritized replay
            alpha (float): Replay sampling, 0 for uniform --> 1 for greedy
            beta_start (float): Compensate for non-uniform sampling.
            beta_end (float): beta ramps linearly. Full compensation at 1.
            beta_episodes (int): Number of episodes to linearly ramp across.
            hidden_layers (list of int): Number of units in hidden layers.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learn_rate = learn_rate
        self.update_every = update_every
        self.replace_every = replace_every
        self.doubleDQN = doubleDQN
        self.soft = soft
        self.duel = duel
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_episodes = beta_episodes
        self.beta_delta = (beta_end - beta_start) / beta_episodes # increments
        self.hidden_layers = hidden_layers

        self.reset()  # reset/initialization (networks, buffer, time, seed)
    
    def reset(self):
        """Initialize networks, replay buffer""" 

        random.seed(self.seed)
        self.t = 0 # time step counter for scheduling and periodic updates 
        
        # Create Q-network
        if self.duel:
            Network = QNet_Duel
        else:
            Network = QNet
        
        self.qnet_local = Network(self.state_size, self.action_size, self.seed, 
                                  self.hidden_layers).to(device)
        self.qnet_target = Network(self.state_size, self.action_size, self.seed,
                                   self.hidden_layers).to(device)

        self.qnet_target.load_state_dict(self.qnet_local.state_dict())      
        self.qnet_target.eval()  # eval mode only for target net        
        self.optimizer = optim.Adam(self.qnet_local.parameters(), 
                                    lr=self.learn_rate)

        # Create experience replay buffer
        if self.prioritized:
            self.replay = ReplayBufferPrioritized(self.action_size, 
                                                  self.buffer_size, 
                                                  self.batch_size, self.seed,
                                                  self.alpha, self.beta_start)
        else:
            self.replay = ReplayBuffer(self.action_size, self.buffer_size, 
                                       self.batch_size, self.seed)
        
    def step(self, state, action, reward, next_state, done):
        """Take step in environment"""

        self.t += 1   # incrememnt time step counter
        # Save experience in replay memory
        self.replay.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps.
        if (self.t % self.update_every) == 0:
            # If enough samples are in memory, get random subset and learn
            if ((self.prioritized) and (self.t > self.batch_size)) or \
              ((not self.prioritized) and (len(self.replay) > self.batch_size)):
                    experiences = self.replay.sample()
                    self.learn(experiences)       
        
        # Increment beta after each episode ends
        if self.prioritized and done:
            if self.replay.beta < self.beta_end:
                self.replay.beta = min(self.beta_end, 
                                       self.replay.beta + self.beta_delta)
                    
    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        
        # Parameters
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def get_targets(self, rewards, next_states):
        """Compute Q-learning targets"""
        with torch.no_grad():  # turn off autograd for targets
            if self.doubleDQN:  
                # Evaluate action values using primary network to find argmax
                self.qnet_local.eval()  # eval mode
                argmaxQ = self.qnet_local.forward(next_states).max(dim=1)[1]\
                                                              .unsqueeze(1)
                self.qnet_local.train()  # turn train back on
                # Select actual max action value w/ argmax on fixed target net
                Q_next = self.qnet_target.forward(next_states)\
                                         .gather(dim=1, index=argmaxQ)       
            else:
                Q_next = self.qnet_target.forward(next_states).max(dim=1)[0]\
                                                              .unsqueeze(1)  
        return rewards + self.gamma * Q_next
                 
    def learn(self, experiences):
        """
        Update network parameters using batch of experience tuples.
        
        # Parameters
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) 
                tuples 
        """
        if self.prioritized:
            states, actions, rewards, next_states, dones, \
                                     weights, batch_idxs = experiences
            weights = weights / max(weights)  # normalized weights
        else:
            states, actions, rewards, next_states, dones = experiences
        
        # Forward pass
        outputs = self.qnet_local.forward(states).gather(dim=1, index=actions)
        
        # Calculate Q-learning target        
        targets = self.get_targets(rewards, next_states)
        
        # Compute loss, do backward pass, take step to minimize loss
        if self.prioritized:
            # Update replay buffer priorities with minibatch TD-errors
            self.replay.update_priorities(batch_idxs, targets, outputs)
            
            criterion = MSE_loss_weighted()
            loss = criterion(outputs, targets, weights)
        else:
            criterion = torch.nn.MSELoss()
            loss = criterion(outputs, targets)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network 
        if self.soft:
            self.soft_update(self.qnet_local, 
                             self.qnet_target, self.tau)
        else:  # hard update
            # Replace target parameters every replace_every time steps.
            if (self.t % self.replace_every) == 0:
                self.hard_update(self.qnet_local, self.qnet_target)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        # Parameters
            local_model (PyTorch model): Weights will be copied from
            target_model (PyTorch model): Weights will be copied to
            tau (float): Interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), 
                                             local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + 
                                    (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        """
        Hard update model parameters.
        θ_target = θ_local

        # Parameters
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), 
                                             local_model.parameters()):
            target_param.data.copy_(local_param.data)

class MSE_loss_weighted(torch.nn.Module):
    """Custom importance-sampling weighted MSE Loss"""
    def __init__(self):
        super(MSE_loss_weighted, self).__init__()
        
    def forward(self, input, target, weights):
        d = weights * (input - target) ** 2
        
        return torch.sum(d) 

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.

        # Paramters
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", 
                                                  "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in 
            experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in 
            experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in 
            experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in 
            experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in 
            experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
    
class ReplayBufferPrioritized:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, 
                 alpha, beta_start):
        """
        Initialize a ReplayBuffer object.

        # Parameters
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (int): Replay sampling, 0 for uniform --> 1 for greedy
            beta_start (int): Starting beta
        """
        self.action_size = action_size
        self.buffer = SumTree(capacity=buffer_size)  
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.p_eps = 0.001
        self.alpha = alpha
        self.beta = beta_start  # to be updated each episode
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", 
                                                  "next_state", "done"])
        self.seed = random.seed(seed)
        self.buffer_used = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory w/ TD-Error for priority."""
        if self.buffer_used < self.buffer_size:
            self.buffer_used += 1  # count buffer 
            
        e = self.experience(state, action, reward, next_state, done)
        
        # Initialize priority to max of current values for future sampling
        if self.buffer.total() == 0:
            p = 1  # initial condition
        else:
            p = np.max(self.buffer.tree[-self.buffer.capacity:])
        # OR TRY: Initialize priority with reward
#        p = (abs(reward) + self.p_eps) ** self.alpha  
        
        self.buffer.add(p, e)
            
    def update_priorities(self, batch_idxs, targets, outputs):
        # Update priorities in replay buffer for transitions learned from in 
        # minibatch             
        td_errors = np.squeeze((targets - outputs).tolist())
        priorities = (abs(td_errors) + self.p_eps) ** self.alpha
        for tree_idx, p_new in zip(batch_idxs, priorities):
            self.buffer.update(tree_idx, p_new)
        
    def get_sampling_wt(self, p):
        """Calculate importance-sampling weight"""
        prob = p / self.buffer.total()
        return (self.buffer_used * prob) ** -self.beta 

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
                               
        experiences = []
        batch_idxs = []
        sampling_wts = []
            
        # Divide sum of priorities into equal ranges for sampling
        p_range = self.buffer.total() / self.batch_size
        
        # Sample a priority value from each range
        for i in range(self.batch_size):
            lo = p_range * i 
            hi = p_range * (i + 1)
            p_sample = random.uniform(lo, hi)
            tree_idx, p, e = self.buffer.get(p_sample)
            batch_idxs.append(tree_idx)
            experiences.append(self.experience(e[0], e[1], e[2], e[3], e[4]))
            # Compute importance-sampling weight to correct for bias
            sampling_wts.append(self.get_sampling_wt(p))

        states = torch.from_numpy(np.vstack([e.state for e in 
            experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in 
            experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in 
            experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in 
            experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in 
            experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        weights = torch.tensor(sampling_wts).float().to(device).unsqueeze(1)
    
        return (states, actions, rewards, next_states, dones, weights, 
                batch_idxs)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer_used)