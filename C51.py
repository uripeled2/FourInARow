from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environment import FourInARow
from evaluation import win_rate

import time
import base64
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()

start = time.time()


# Hyperparameters

num_iterations = 15000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

fc_layer_params = (100,)

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = num_iterations // 20  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = -10  # @param {type:"integer"}
max_q_value = 1  # @param {type:"integer"}
n_step_update = 15  # @param {type:"integer"}

num_eval_episodes = 100  # @param {type:"integer"}
eval_interval = num_iterations // 20  # @param {type:"integer"}


# setup the env

train_py_env = FourInARow()
eval_py_env = FourInARow()

# convert the env to tf_env
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


# Agent

# setup the categorical network
categorical_q_net = categorical_q_network.CategoricalQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    num_atoms=num_atoms,
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network=categorical_q_net,
    optimizer=optimizer,
    min_q_value=min_q_value,
    max_q_value=max_q_value,
    n_step_update=n_step_update,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=train_step_counter)
agent.initialize()


# setup the policies
eval_policy = agent.policy  # The main policy that is used for evaluation and deployment
collect_policy = agent.collect_policy  # A second policy that is used for data collection
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())


# Data Collection

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)


for _ in range(initial_collect_steps):
    collect_step(train_env, random_policy)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)


# Training

agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
wins = win_rate(eval_env, agent.policy, num_eval_episodes)
print('step = {0}: Wining rate = {1:.2f}'.format(0, wins))
returns = [wins]

for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent.collect_policy)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            wins = win_rate(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Wining rate = {1:.2f}'.format(step, wins))
            returns.append(wins)


print('sec:', time.time() - start)

# Visualization

# Run a game
environment = eval_env
policy = agent.policy
time_step = environment.reset()

while not time_step.is_last():
    print(time_step.observation)
    action_step = policy.action(time_step)
    time_step = environment.step(action_step.action)
print('last:', time_step.observation)

# Plot
iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Wining rate')
plt.xlabel('Iterations')
plt.ylim(top=100)
plt.show()
