# Deep Reinforcement Learning for Navigation

Train an agent to navigate the Banana Collectors environment from Unity ML-Agents using Deep Q-Networks (DQN) with dueling architecture and prioritized experience replay. 

Watch a video of the agent here: [https://youtu.be/8ec-FtBbxic](https://youtu.be/8ec-FtBbxic)

![Banana Collectors Environment](assets/banana_world.png)

## The  Environment

The Banana Collectors environment is a large flat square world enclosed by walls. Yellow and blue bananas are scattered throughout the environment. Four discrete actions are available to the agent: move forward, move backward, turn left, and turn right. The agent senses the environmental state through ray-based perception of the objects in the forward direction and through velocity. These combine for a state space with 37 dimensions. Navigating to a yellow banana provides a reward of +1, while a blue banana provides a negative reward of -1. Each episode allows the agent 300 steps before ending. The criteria for solving the task is averaging a score of 13 points across 100 episodes. 

## Installation

1. Create and activate a  Python 3.6 environment. Choose an environment name in place of `your_name`.
```bash
conda create -n your_name python=3.6
source activate your_name
```

2. Create an IPython kernel for your new environment.
```bash
python -m ipykernel install --user --name your_name --display-name "your_name"
```

3. Clone this repository and install dependencies in the `python/` folder, which comes from the [Udacity Deep Reinforcement Learning](https://github.com/udacity/deep-reinforcement-learning) repository. These dependencies will include PyTorch and Unity ML-Agents Toolkit.
```bash
git clone https://github.com/supercurious/deep-rl-navigation.git
cd python
pip install .
```

4. Download the Unity environment and unzip the file.
    * [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    * [Linux (headless version for training on AWS)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)
    * [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    * [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## Usage

1. Open the Jupyter notebook `REPORT.ipynb` for implementation and results.
```bash
jupyter notebook REPORT.ipynb
```

2. From the top menu bar, click on "Kernel", navigate to "Change kernel" and select the new environment you created during installation.