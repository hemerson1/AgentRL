# AgentRL

A series of ready-to-use reinforcement learning agents with a focus on easy integration with custom learning environments. 

## Contents

The following reinforcement learning algorithms have been currently implemented:

### Rainbow DQN

**File**: ```deep_q_network.py```

**Description**: This algorithm extends the traditional deep Q network approach to incorporate six additional features:
* Double Q network - uses two seperate deep Q networks to perform action selection and evaluation.
* Prioritsed Replay - prioritises samples with a larger difference between their expected and observed loss. 
* Dueling Networks - seperates the estimation of state values and the state-action advantage.
* Multi-step - consider the reward over multiple steps.
* Distributional Learning - approximate the return distribution rather than the expected value.  
* Noisy Networks - introduce noise to neural network linear layers.

**Reference Paper**: https://arxiv.org/abs/1710.02298

## Installation

It is recommended to install the library using pip, preferrably within a virtual environment to avoid dependency conflicts. 

```
pip install AgentRL
```

## Usage

Here is an example of the Rainbow DQN being used on the CartPole environment available from OpenAI Gym (https://github.com/openai/gym).

```python
import gym

from AgentRL.agents import DQN
from AgentRL.common.buffers import prioritised_replay_buffer

# initialise the environment 
env = gym.make("CartPole-v0")

# initialise the buffer
buffer = prioritised_replay_buffer()

# initialise the agent
agent = DQN(
    state_dim=4,action_num=2, 
    replay_buffer=buffer,
    algorithm_type='rainbow',
    multi_step=3
)

for ep in range(episodes):
    state, done = env.reset(), False
    while not done:

        # take an action and update the environment
        action = agent.get_action(state)              
        next_state, reward, done, info = env.step(int(action))
        
        # push samples to the replay buffer
        agent.push(state, action, next_state, reward, done)
        agent.update()                       

        # update the state
        state = next_state
```

## Licence
[MIT](https://choosealicense.com/licenses/mit/)