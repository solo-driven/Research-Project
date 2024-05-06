from DeepQNetwork import DeepQNetwork
import numpy as np
import torch as T

class Agent():
    def __init__(self, discount_factor, exploration_rate, learning_rate, input_dimensions, batch_size, num_actions=3,
                 max_memory_size=10000000, min_exploration_rate=1e-4, exploration_rate_decay=1e-4, first_layer_dimensions=256, second_layer_dimensions=128, third_layer_dimensions=64):
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.action_space = [i for i in range(num_actions)]
        self.memory_size = max_memory_size
        self.batch_size = batch_size
        self.memory_counter = 0

        # Initialize the Q-Network
        self.Q_network = DeepQNetwork(self.learning_rate, input_dims=input_dimensions, n_actions=self.num_actions,
                                   fc1_dims=first_layer_dimensions, fc2_dims=second_layer_dimensions, fc3_dims=third_layer_dimensions)

        # Initialize memory for states, new states, actions, rewards and terminal states
        self.state_memory = np.zeros((self.memory_size, *input_dimensions), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_dimensions), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)

        self.loss = None

    def save_model(self):
        # Save the current state of the Q-Network
        T.save(self.Q_network, 'DNN_Params')

    def load_model(self):
        # Load the saved state of the Q-Network
        self.Q_network = T.load('DNN_Params')

    def store_transition(self, state, action, reward, new_state, done):
        # Store the transition in memory
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def choose_action(self, observation):
        # Choose an action based on the current state
        if np.random.random() > self.exploration_rate:
            state = T.tensor(observation).to(self.Q_network.device)
            actions = self.Q_network.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        # Learn from the memory
        if self.memory_counter < self.batch_size:
            return

        # Initialize the gradient
        self.Q_network.optimizer.zero_grad()

        # Return the size of memory
        max_memory = min(self.memory_counter, self.memory_size)

        # Randomize a batch
        batch = np.random.choice(max_memory, self.batch_size, replace=False)

        # Array with index
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # extract a batch from the memory
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_network.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_network.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_network.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_network.device)

        action_batch = self.action_memory[batch]

        # Calculate Q values for current and next states
        q_current = self.Q_network.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_network.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        # Calculate target Q value
        q_target = reward_batch + self.discount_factor * T.max(q_next, dim=1)[0]

        # Calculate loss
        loss = self.Q_network.loss(q_target, q_current).to(self.Q_network.device)
        loss.backward()
        self.loss = self.Q_network.loss(q_target, q_current).detach().numpy()

        # Update the Q-Network
        self.Q_network.optimizer.step()

        # Update the exploration rate
        self.exploration_rate = self.exploration_rate - self.exploration_rate_decay if self.exploration_rate > self.min_exploration_rate else self.min_exploration_rate