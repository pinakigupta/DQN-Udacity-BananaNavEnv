[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)

    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    - This project uses Unity's rich environments to design, train, and evaluate deep reinforcement learning algorithms. **To run this project you'll need to install Unity ML-Agents.**You can read more about ML-Agents and how to install it by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents). 

      > **Note: The Unity ML-Agent team frequently releases updated versions of their environment. We are using the v0.4 interface. To avoid any confusion, please use the workspace we provide here or work with v0.4 locally.**

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Code Description

1. Navigation.ipynb - Main module containing 1)loading of the helper modules 2) loading of the DQN agent helper module 3)training the DQN agent 4)plotting results and 5) checkpointing the model parameters.
2. model.py - loads pytroch module and derives a custom NN model for this problem
3. dqn_agent.py - Helper module contains 1) loads the helper model.py module 2)uses the NN model to train a DQN agent 3) Containts prioritized(optional) experience replay buffer from which the DQN draws sample 


### Important Hyperparameters 

1. Navigation.ipynb - Main Module contains most of the hyper parameters for training the DQN agent 
		a. n_episodes. Maximum number of episodes for which training will proceed
		b. max_t. maximum number of steps per episode during training
		c. eps_start, eps_end, eps_decay - During the exploration using an episilon greedy policy is used. The policy starts with eps_start at episode 1 and decays by eps_decay each episode
		until it hits the eps_end floor.
		d. random_replay - If True random sampling of experience buffer is chosen, otherwise the prioritized sampling of experience buffer is chosen
		e. dqn_fc_layer - architecture of the Hidden layers of the Q network. ex. = [ 64 64 32 256] means there are 4 hidden layers of 64, 64, 32 and 256 units, in that order.
		f. checkpoint_score - if the score is greater than this threshold, every 100 episode the network will be checkpointed. This can be set as a score 
		target for a reasonably good agent.
		g. breakpoint_score - if the score is greater than this threshold, network is checkpointed and training is finished. This can be set as a score 
		target for a exceptionally good agent.
2. ddpg_agent.py -   

		BUFFER_SIZE = int(1e5)  # replay buffer size
		BATCH_SIZE = 64         # minibatch size
		GAMMA = 0.99            # discount factor
		TAU = 1e-3              # for soft update of target parameters
		LR = 5e-4               # learning rate 
