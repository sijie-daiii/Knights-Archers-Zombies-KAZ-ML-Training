# CS4100 Knights Archers Zombies (KAZ) Centralized and Decentralized Training
Knights Archers Zombies is a game provided by the ButterFly enviroment in Petting Zoo. The game consists of archers and knights fighting off zombies that are coming in waves. The video below shows an example of the game where the agents are taking random actions.

![](https://github.com/parnikajain/CS4100KAZ/blob/main/butterfly_knights_archers_zombies.gif)

[Petting Zoo](https://pettingzoo.farama.org/content/basic_usage/) in general offers a variety of multi-agent enviroments to train and test different Reinforcement Learning algorithms. Our goal was to implement Decentralized and Centralized training to the archers and knights in hope to see optimal performance. For Decentralized Training, we implemented Deep Q Networks where each agent updates its own policy without considering the behavior of the other agents in the enviroment. On the other hand, for Centralized Training, we implemented a traditional MLP neural network architecture with action pooling which allowed all agents to share information and coordinate their actions during training, updating a joint policy that considers the behavior of all agents in the environment. 

## Resources
If you are new to Reinforcement Learning or would like some clarification regarding our implementation, no problem! Here we have a list of resources that we found useful for our implementation of Decentralized and Centralized Training in a Multi-Agent Enviroment.

- [Intro to RL and Q network tutorial](https://www.tensorflow.org/agents/tutorials/0_intro_rl)
- [QMIX Research Paper](https://arxiv.org/pdf/1803.11485)
- [Article on Centralized Training and Decentralized Execetion in MARL](https://blog.devops.dev/centralised-training-and-decentralised-execution-in-multi-agent-reinforcement-learning-e68535a05307)

## Installations 
Before you decide to train or execute the pre-trained models, you need to install the necessary dependencies. Also, Petting Zoo states that only Python 3.8-3.11 is supported on macOS and Linux so make sure you update your Python version if you haven't!

```bash
pip install numpy matplotlib torch tianshou pettingzoo['butterfly']
```
- numpy: Large, multi-dimensional arrays and matrices
- matplotlib: Data visualizations for reward and loss plots
- pytorch: Deep learning library for building and training neural networks
- tianshou: Reinforcement learning library with various algorithms and tools
- pettingzoo: Petting zoo's Butterfly enviroment with multi-agent games for cooperative and competitive scenarios

## Decentralized Training
### Github Clone
First, clone the repo
```bash
git clone https://github.com/parnikajain/CS4100KAZ.git
```

Next, move into the project directory and into the Decentralized_Training directory to access the model
```bash
cd CS4100KAZ
cd Decentralized_Training
```

### Training
To continue to train our premodel in a decentralized setting, run the following command:

```bash
python model.py --mode train
```

### Evaluation
To evaluate the trained model, you can run the following command:

```bash
python model.py --mode eval
```
### Notes
- After training, a visualization of the loss and training rewards will show up
- After evaluation, a visualization of the total rewards will show up along with the average reward in the console.

## Centralized Training
### Training
To start training in a centralized setting, run the following command:

```bash
python KAZ_centralize_train.py
```

### Visualization
To observe the agents' performance during Centralized training:

```bash
python KAZ_centralize_train.py --observe_only
```

### Data Viewing
You can monitor the training progress and metrics using TensorBoard:

```bash
tensorboard --logdir=data
```

### Notes
- **Visualization**: The `--observe_only` flag lets you watch the trained agents in action without further training, providing insight into their learned behaviors.
- **TensorBoard**: Use TensorBoard for a detailed view of training metrics, such as loss curves and reward progress, to evaluate the effectiveness of centralized training.
