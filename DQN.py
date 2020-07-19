from __future__ import absolute_import, division, print_function
from environment import FourInARow

import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()


# Hyperparameters

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = num_iterations // 20  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = num_iterations // 20  # @param {type:"integer"}

fc_layer_params = (100,)  # A tuple describing the number and size of the model's hidden layers


# setup the env
train_py_env = FourInARow()
eval_py_env = FourInARow()

# convert the env to tf_env
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


# setup the net
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()


# setup the policies
eval_policy = agent.policy  # The main policy that is used for evaluation and deployment
collect_policy = agent.collect_policy  # A second policy that is used for data collection
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())


# Metrics and evaluation
def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    tensor_avg_return = total_return / num_episodes
    num_avg_return = tensor_avg_return.numpy()[0]
    return num_avg_return


def win_rate(environment, policy, num_episodes=10):
    wins = 0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        if episode_return > 30:
            wins += 1
    return (wins / num_episodes) * 100


# setup replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=agent.collect_data_spec,
                batch_size=train_env.batch_size,
                max_length=replay_buffer_max_length)


# Data collection
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


collect_data(train_env, random_policy, replay_buffer, steps=100)

# DQN Agent needs both the current and next observation to compute the loss so the \
# dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
          num_parallel_calls=3,
          sample_batch_size=batch_size,
          num_steps=2).prefetch(3)

iterator = iter(dataset)


# Training the agent

# Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
# avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
wins = win_rate(eval_env, agent.policy, num_eval_episodes)
returns = [wins]

for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent.collect_policy, replay_buffer)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print(F'step = {step}: loss = {train_loss}')

    if step % eval_interval == 0:
        # avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        wins = win_rate(eval_env, agent.policy, num_eval_episodes)
        # print(F'step = {step}: Average Return = {wins}')
        print(F'step = {step}: Wining rate = {wins}')
        returns.append(wins)


# Visualization

# Run a game

environment = eval_env
policy = eval_policy
time_step = environment.reset()

while not time_step.is_last():
    print(time_step.observation)
    action_step = policy.action(time_step)
    time_step = environment.step(action_step.action)
print('last:', time_step.observation)

# Plot

iterations = range(0, num_iterations + 1, eval_interval)
print('iter:', iterations)
print('ret:', returns)
plt.plot(iterations, returns)
plt.ylabel('Wining rate')
plt.xlabel('Iterations')
plt.ylim(top=100)
plt.show()




