# https://www.youtube.com/watch?v=vcGv5vmOydc

"""
This code produces tf version errors that have not yet been resolved!
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.initializers import random_uniform
from copy import deepcopy
import gym


class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x = x
        return x


# Class for replay buffer
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros_like(self.state_memory)
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, is_done):
        """
        Adds a transition to the replay buffer
        """
        index = self.mem_ctr % self.mem_size  # wraps around to the beginning of buffer when out of space
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - is_done
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        """
        Get samples from the replay buffer
        """
        max_mem = min(self.mem_size, self.mem_ctr)  # if mem_ctr < mem_size, sample from inside [0, mem_ctr]
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal_states = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminal_states


# Actor class
class Actor:
    def __init__(self, learning_rate, number_of_actions, network_name, input_dimensions, session,
                 fc1_dimensions, fc2_dimensions, action_bound, batch_size=64, checkpoint_dir='tmp/ddpg'):
        self.learning_rate = learning_rate
        self.number_of_actions = number_of_actions
        self.network_name = network_name
        self.input_dimensions = input_dimensions
        self.session = session
        self.fc1_dimensions = fc1_dimensions
        self.fc2_dimensions = fc2_dimensions
        self.action_bound = action_bound
        self.batch_size = batch_size

        self.params = tf.trainable_variables(scope=self.network_name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(checkpoint_dir, self.network_name + '_ddpg.ckpt')
        self.build_network()

        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # gradient of network wrt parameters
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients,
                                                                                        self.params))

        # self.saver = tf.train.Saver()

    def build_network(self):
        with tf.variable_scope(self.network_name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dimensions],
                                        name='inputs')
            self.action_gradient = tf.placeholder(tf.float32,
                                                  shape=[None, self.number_of_actions],
                                                  name='gradients')

            # 1st layer
            f1 = 1. / np.sqrt(self.fc1_dimensions)
            dense1 = tf.layers.dense(self.input,
                                     units=self.fc1_dimensions,
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)  # can be done before/after activation
            layer1_activation = tf.nn.relu(batch1)

            # 2nd layer
            f2 = 1. / np.sqrt(self.fc2_dimensions)
            dense2 = tf.layers.dense(layer1_activation,
                                     units=self.fc2_dimensions,
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)  # can be done before/after activation
            layer2_activation = tf.nn.relu(batch2)

            # 3rd layer (output layer)
            f3 = 0.003
            mu = tf.layers.dense(layer2_activation,
                                 units=self.number_of_actions,
                                 activation='tanh',  # bounding action spaces between [-1, 1]
                                 kernel_initializer=random_uniform(-f3, f3),
                                 bias_initializer=random_uniform(-f3, f3))
            self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs):
        """
        Given an observation, predict an action
        """
        return self.session.run(self.mu, feed_dict={self.input: inputs})

    def train_network(self, inputs, gradients):
        self.session.run(self.optimizer, feed_dict={self.input: inputs,
                                                    self.action_gradient: gradients})

    def load_checkpoint(self):
        """
        [book-keeping function]
        Loads session from checkpoint file and sets it on to the current session
        """
        print("load_checkpoint()")
        self.saver.restore(self.session, self.checkpoint_file)

    def save_checkpoint(self):
        """
        [book-keeping function]
        Save current session in the checkpoint file
        """
        print("save_checkpoint()")
        self.saver.save(self.session, self.checkpoint_file)


# Critic class
class Critic:
    def __init__(self, learning_rate, number_of_actions, network_name, input_dimensions, session,
                 fc1_dimensions, fc2_dimensions, batch_size=64, checkpoint_dir='tmp/ddpg'):
        self.learning_rate = learning_rate
        self.number_of_actions = number_of_actions
        self.network_name = network_name
        self.input_dimensions = input_dimensions
        self.session = session
        self.fc1_dimensions = fc1_dimensions
        self.fc2_dimensions = fc2_dimensions
        self.batch_size = batch_size

        self.params = tf.trainable_variables(scope=self.network_name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(checkpoint_dir, self.network_name + '_ddpg.ckpt')
        self.build_network()

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)  # minimize critic's loss
        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.variable_scope(self.network_name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dimensions],
                                        name='inputs')
            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, self.number_of_actions],
                                          name='actions')
            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None, 1],
                                           name='targets')

            # 1st layer
            f1 = 1. / np.sqrt(self.fc1_dimensions)
            dense1 = tf.layers.dense(self.input,
                                     units=self.fc1_dimensions,
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)  # can be done before/after activation
            layer1_activation = tf.nn.relu(batch1)

            # 2nd layer
            f2 = 1. / np.sqrt(self.fc2_dimensions)
            dense2 = tf.layers.dense(layer1_activation,
                                     units=self.fc2_dimensions,
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)  # can be done before/after activation

            # 3rd layer, feeding in the actions
            action_in = tf.layers.dense(self.actions,
                                        units=self.fc2_dimensions,
                                        activation='relu')

            # 4th layer, feed in both actions and states
            state_actions = tf.nn.relu(tf.add(batch2, action_in))

            # 5th layer (last layer)
            f3 = 0.003
            self.q = tf.layers.dense(state_actions,
                                     units=1,  # action-value function is a scalar
                                     kernel_initializer=random_uniform(-f3, f3),
                                     bias_initializer=random_uniform(-f3, f3),
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01))  # 0.01 lifted from paper
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.session.run(self.q,
                                feed_dict={self.input: inputs,
                                           self.actions: actions})

    def train(self, inputs, actions, q_target):
        return self.session.run(self.optimizer,
                                feed_dict={self.input: inputs,
                                           self.actions: actions,
                                           self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.session.run(self.action_gradients,
                                feed_dict={self.input: inputs,
                                           self.actions: actions})

    def load_checkpoint(self):
        """
        [book-keeping function]
        Loads session from checkpoint file and sets it on to the current session
        """
        print("load_checkpoint()")
        self.saver.restore(self.session, self.checkpoint_file)

    def save_checkpoint(self):
        """
        [book-keeping function]
        Save current session in the checkpoint file
        """
        print("save_checkpoint()")
        self.saver.save(self.session, self.checkpoint_file)


# Agent class
class Robot:
    def __init__(self,
                 alpha,  # learning rate for actor network
                 beta,  # learning rate for critic network
                 input_dimensions,
                 tau,
                 env,  # environment required to get action bounds
                 gamma=0.99,
                 number_of_actions=2,
                 max_size=1000000,  # size of replay buffer
                 layer1_size=400,  # from the paper
                 layer2_size=300,  # from the paper
                 batch_size=64,
                 checkpoint_directory='tmp/ddpg'):
        self.gamma = gamma  # discount factor for Bellman equation
        self.tau = tau  # multiplicative factor for SOFT update of network parameters
        self.memory = ReplayBuffer(max_size=max_size, input_shape=input_dimensions, n_actions=number_of_actions)
        self.batch_size = batch_size
        self.session = tf.Session()  # instantiate single session and pass it to all 4 networks

        self.actor = Actor(learning_rate=alpha,
                           number_of_actions=number_of_actions,
                           network_name='Actor',
                           input_dimensions=input_dimensions,
                           session=self.session,
                           fc1_dimensions=layer1_size,
                           fc2_dimensions=layer2_size,
                           action_bound=env.action_space.high,
                           checkpoint_dir=checkpoint_directory)
        self.critic = Critic(learning_rate=beta,
                             number_of_actions=number_of_actions,
                             network_name='Critic',
                             input_dimensions=input_dimensions,
                             session=self.session,
                             fc1_dimensions=layer1_size,
                             fc2_dimensions=layer2_size,
                             checkpoint_dir=checkpoint_directory)
        self.target_actor = Actor(learning_rate=alpha,
                                  number_of_actions=number_of_actions,
                                  network_name='TargetActor',
                                  input_dimensions=input_dimensions,
                                  session=self.session,
                                  fc1_dimensions=layer1_size,
                                  fc2_dimensions=layer2_size,
                                  action_bound=env.action_space.high,
                                  checkpoint_dir=checkpoint_directory)
        self.target_critic = Critic(learning_rate=beta,
                                    number_of_actions=number_of_actions,
                                    network_name='TargetCritic',
                                    input_dimensions=input_dimensions,
                                    session=self.session,
                                    fc1_dimensions=layer1_size,
                                    fc2_dimensions=layer2_size,
                                    checkpoint_dir=checkpoint_directory)
        self.noise = OUActionNoise(mu=np.zeros(number_of_actions))

        # Placing these outsides outside __init__ slows down execution
        self.update_actor = [self.target_actor.params[i].assign(tf.multiply(self.actor.params[i], self.tau) +
                                                                tf.multiply(self.target_actor.params[i],
                                                                            1.0 - self.tau))
                             for i in range(len(self.target_actor.params))]
        self.update_critic = [self.target_critic.params[i].assign(tf.multiply(self.critic.params[i], self.tau) +
                                                                  tf.multiply(self.target_critic.params[i],
                                                                              1.0 - self.tau))
                              for i in range(len(self.target_critic.params))]

        self.session.run(tf.global_variables_initializer())
        self.update_network_parameters(first_time=True)

    def update_network_parameters(self, first_time=False):
        if first_time:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.session.run(self.update_critic)
            self.target_actor.session.run(self.update_actor)
            self.tau = old_tau

        else:
            self.target_critic.session.run(self.update_critic)
            self.target_actor.session.run(self.update_actor)

    def add_to_replay_buffer(self, state, action, reward, new_state, is_done):
        """
        Store transitions in replay buffer memory
        """
        self.memory.store_transition(state, action, reward, new_state, is_done)

    def choose_action(self, state):
        state_local_copy = deepcopy(state)
        state_local_copy = state_local_copy[np.newaxis, :]
        mu = self.actor.predict(state_local_copy)  # list of lists
        mu_prime = mu + self.noise()
        return mu_prime[0]

    def learn(self):
        """
        Sample from replay buffer only after it has become full
        """

        if self.memory.mem_ctr < self.batch_size:
            return

        state, action, reward, new_state, is_done = self.memory.sample_buffer(self.batch_size)

        # Two feedforward predictions chained
        critic_value = self.target_critic.predict(inputs=new_state,
                                                  actions=self.target_actor.predict(inputs=new_state))

        # Calculating q-targets using non-vectorized method to avoid batch_size X batch_size dimensions
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value[j] * is_done[j])
        target = np.reshape(target, (self.batch_size, 1))  # self.q_target has shape batch_size X 1

        # Critic training
        _ = self.critic.train(inputs=state, actions=action, q_target=target)

        action_predicted = self.actor.predict(inputs=state)
        gradients = self.critic.get_action_gradients(inputs=state, actions=action_predicted)
        self.actor.train_network(inputs=state, gradients=gradients[0])

        # After training actor and critic, update network parameters
        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()


if __name__ == '__main__':

    env = gym.make('LunarLanderContinuous-v2')
    agent = Robot(alpha=0.0001, beta=0.001,
                  input_dimensions=env.observation_space.shape, tau=0.001,
                  batch_size=64, layer1_size=400, layer2_size=300,
                  number_of_actions=env.action_space.shape[0],
                  env=env)

    n_games = 1000

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.add_to_replay_buffer(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)

    # robot = Robot(alpha=1e-4, beta=1e-3, input_dimensions=[24], tau=1e-3, env=env, number_of_actions=4)
    #
    # np.random.seed(0)
    #
    # score_history = []
    # for i in range(1500):   # number of games
    #     observation = env.reset()
    #     done = False
    #     score = 0
    #     while not done:
    #         action = robot.choose_action(observation)
    #         new_state, reward, done, info = env.step(action=action)
    #         robot.add_to_replay_buffer(state=observation, action=action, reward=reward,
    #                                    new_state=new_state, is_done=int(done))
    #         robot.learn()
    #         score += reward
    #
    #         observation = new_state
    #
    #         env.render()
    #
    #     score_history.append(score)
    #     print("Episode %i: score: %.3f" % (i, score))
    #
    #     # Save every 25 games
    #     if i % 25 == 0:
    #         robot.save_models()
