"""
This script takes command line arguments and trains a DQN agent to navigate the 
Banana Collectors environment in Unity ML-Agents.

It is meant to be called from a shell script, which specifies and loops through 
two arguments: the agent random seed and the agent configuration number.

Looping through different agent random seeds provides a statistical look at 
learning performance with different random initializations.

The agent configuration number is defined as follows:
0: DQN
1: Double DQN
2: Double DQN Dueling
3: Double DQN PER
4: Double DQN PER + Dueling

So why loop through training runs in the shell rather than in the python 
script? For a fair comparison between configurations, a common environment 
random seed is desired to eliminate variance in environment conditions. 
Closing and re-instantiating the environment should allow a reset of the 
environment seed. Currently, however, the Unity environment cannot be 
re-instantiated and requires a kernal restart. I ran into this error in both 
OSX and Linux. Moreover, I couldn't find a way to programmatically restarted 
the kernal from within the Jupyter notebook.

As a workaround, I use a shell script to manage each training run by calling 
this Python script and passing in arguments for the agent seed and configuration. 
This approach ensures the environment seed is the same for every training run.

"""

# Imports
import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline
import random
import torch
from collections import deque
from unityagents import UnityEnvironment
from dqn_agent import Agent
import time
import pickle
import os
import sys

# Take command line arguments to specify agent random seed and configuration
seed_agent = int(sys.argv[1]) # random seed for agent
config = int(sys.argv[2])  # agent configuration number

print("################################")
print("agent_seed = {} (sys.argv[1])".format(seed_agent))
print("config     = {} (sys.argv[2])".format(config))
print("################################")

# Start environment
#env = UnityEnvironment(file_name="Banana.app", seed=10)  # run on macbook
env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64", 
                       seed=10)  # run headless on AWS

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Examine state and action spaces
env_info = env.reset(train_mode=True)[brain_name]
print('Number of agents:', len(env_info.agents))
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# Train DQN agent
def train_dqn(n_episodes=500, max_t=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.98):
    """Train DQN Agent
    
    # Parameters 
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    solved = False
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
                
        if agent.prioritized:
            print('\rEpisode {}\tAverage Score: {:.2f}\teps: {:.4f}\tBeta: {:.4f}'
                  .format(i_episode, np.mean(scores_window), eps, agent.replay.beta), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}\teps: {:.4f}\tBeta: {:.4f}'
                      .format(i_episode, np.mean(scores_window), eps, agent.replay.beta))
        if not agent.prioritized: 
            print('\rEpisode {}\tAverage Score: {:.2f}\teps: {:.4f}'
                  .format(i_episode, np.mean(scores_window), eps), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}\teps: {:.4f}'
                      .format(i_episode, np.mean(scores_window), eps))
        if (np.mean(scores_window)>=13) and not solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnet_local.state_dict(), 'checkpoint.pth')
            solved = True
            #break

    return scores

def make_label(agent):
    """Make label based on agent configuration"""
    
    label = "double{}_duel{}_per{}_A{}_Bstart{}_Bend{}_Bep{}_LR{}_layers{}".format(
            agent.doubleDQN, agent.duel, agent.prioritized , agent.alpha, 
            agent.beta_start, agent.beta_end, agent.beta_episodes, agent.learn_rate,
            agent.hidden_layers)
    return label

def save_logs(scores, label, logs={}):
    """Log scores and save"""
    
    logs[label] = scores

    #results_dir = "results/"
    results_dir = "results_aws/"  # training on AWS
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)  

    timestamp = time.strftime("%H%M%S")
    filename = label + "_" + timestamp + ".pkl"
    with open(results_dir + filename, 'wb') as f:
        pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)
    print("Logs saved for: \n" + label + "\n")
    
    return logs

def stat_runs(agent, n_runs=1):
    """Run once or multiple times for statistics"""
    
    logs={}
    label = make_label(agent)

    # Train agent
    for i in range(n_runs):
        if i > 0:
            agent.reset()  # reset weights, buffer, seed
        t0 = time.time()
        scores = train_dqn()
        t1 = time.time()
        print("\nRun time: {:.2f} minutes.".format((t1-t0)/60))
        
        #plot_score(scores) # plot score
        
        # Do logs
        label = make_label(agent)
        logs = save_logs(scores, label, logs)
    
    return logs

# Configure agent and train
n_runs = 1

if config==0:
    # Create agent: DQN
    agent = Agent(state_size=37, action_size=4, seed=seed_agent, soft=True, 
                  doubleDQN=False, duel=False, prioritized=False,
                  alpha=0, beta_start=0, beta_end=0, beta_episodes=200,
                  hidden_layers=[128,64], learn_rate=5e-4)
    print("Config: " + make_label(agent))
    # Run training simulation(s)
    logs_DQN = stat_runs(agent, n_runs)
    
elif config==1:
    # Create agent: Double DQN
    agent = Agent(state_size=37, action_size=4, seed=seed_agent, soft=True, 
                  doubleDQN=True, duel=False, prioritized=False,
                  alpha=0, beta_start=0, beta_end=0, beta_episodes=200,
                  hidden_layers=[128,64], learn_rate=5e-4)
    print("Config: " + make_label(agent))
    # Run training simulation(s)
    logs_doubleDQN = stat_runs(agent, n_runs)
    
elif config==2:
    # Create agent: Double DQN Dueling
    agent = Agent(state_size=37, action_size=4, seed=seed_agent, soft=True, 
                  doubleDQN=True, duel=True, prioritized=False,
                  alpha=0, beta_start=0, beta_end=0, beta_episodes=200,
                  hidden_layers=[128,64], learn_rate=5e-4)
    print("Config: " + make_label(agent))
    # Run training simulation(s)
    logs_doubleDQN_duel = stat_runs(agent, n_runs)
    
elif config==3:
    # Create agent: Double DQN PER
    agent = Agent(state_size=37, action_size=4, seed=seed_agent, soft=True, 
                  doubleDQN=True, duel=False, prioritized=True,
                  alpha=0.6, beta_start=0.4, beta_end=1, beta_episodes=200,
                  hidden_layers=[128,64], learn_rate=5e-4/2)
    print("Config: " + make_label(agent))
    # Run training simulation(s)
    logs_doubleDQN_PER = stat_runs(agent, n_runs)
    
elif config==4:
    # Create agent: Double DQN PER + Dueling
    agent = Agent(state_size=37, action_size=4, seed=seed_agent, soft=True, 
                  doubleDQN=True, duel=True, prioritized=True,
                  alpha=0.6, beta_start=0.4, beta_end=1, beta_episodes=200,
                  hidden_layers=[128,64], learn_rate=5e-4/2)
    print("Config: " + make_label(agent))
    # Run training simulation(s)
    logs_doubleDQN_duel_PER = stat_runs(agent, n_runs)