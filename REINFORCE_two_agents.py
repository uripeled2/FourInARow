
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environment import FourInARow
from evaluation import win_rate, compute_avg_return

import time
import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import os
import tempfile
import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver, random_tf_policy

tf.compat.v1.enable_v2_behavior()

start = time.time()

# Hyperparameters

num_iterations: int = 1000
collect_episodes_per_iteration: int = 2
replay_buffer_capacity: int = 2000

fc_layer_params: tuple = (100,)

learning_rate: float = 1e-3
log_interval: int = 100
num_eval_episodes: int = 100
eval_interval: int = 100

rounds: int = 100
wining_rate_goal: int = 78

save_policy1: str = "player1-0.0"
save_policy2: str = "player2-0.0"

# setup the env
def crete_envs(policy,  rest: bool = True):
    train_py_env = FourInARow(policy, rest)
    eval_py_env = FourInARow(policy, rest)

    # convert the env to tf_env
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    return train_env, eval_env

dummy_env, _ = crete_envs(None, False)
enemy_policy = random_tf_policy.RandomTFPolicy(dummy_env.time_step_spec(), dummy_env.action_spec())
train_env, eval_env = crete_envs(enemy_policy)


# setup agents
def create_agent():
    # setup the net
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)

    tf_agent.initialize()
    return tf_agent

tf_agent1 = create_agent()
tf_agent2 = create_agent()

# Policies
eval_policy1 = tf_agent1.policy
collect_policy1 = tf_agent1.collect_policy
eval_policy2 = tf_agent2.policy
collect_policy2 = tf_agent2.collect_policy

# Replay buffers
replay_buffer1 = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent1.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)
replay_buffer2 = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent1.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)


# Data Collection
def collect_episode(environment, policy, replay_buffer, num_episodes):
    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1


# Training

tf_agent1.train = common.function(tf_agent1.train)
tf_agent2.train = common.function(tf_agent2.train)

# Reset the train step
tf_agent1.train_step_counter.assign(0)
tf_agent2.train_step_counter.assign(0)

itrs = []
itr = 0
rou = 1

def play_until_you_win(tf_agent, replay_buffer, target: int = wining_rate_goal):
    wining_rate = win_rate(eval_env, tf_agent.policy, num_eval_episodes)
    print("Start wining_rate =", wining_rate)
    while wining_rate < target:
        global itr
        itr += 1
        # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(
            train_env, tf_agent.collect_policy, replay_buffer, collect_episodes_per_iteration)

        # Use data from the buffer and update the agent's network.
        experience = replay_buffer.gather_all()
        train_loss = tf_agent.train(experience)
        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            wining_rate = win_rate(eval_env, tf_agent.policy, num_eval_episodes)
            print(F'step = {step}: Wining rate = {wining_rate}')


# Play ageist yourself
for i in range(rounds):
    one = i % 2 == 0
    if one:
        print("Agent1:")
        play_until_you_win(tf_agent1, replay_buffer1)
    else:
        print("Agent2:")
        play_until_you_win(tf_agent2, replay_buffer2)

    print(F'Finished round {rou} in: {time.time() - start}, num of itr = {itr}')
    itrs.append(itr)
    itr = 0
    rou += 1
    if i != rounds - 1:
        # Update env
        if one:
            enemy_policy = tf_agent1.policy
            train_env, eval_env = crete_envs(enemy_policy)
        else:
            enemy_policy = tf_agent2.policy
            train_env, eval_env = crete_envs(enemy_policy)

print(F"Total time = {time.time() - start}")

# Save
tf_policy1_saver = policy_saver.PolicySaver(tf_agent1.policy)
tf_policy2_saver = policy_saver.PolicySaver(tf_agent2.policy)
tf_policy1_saver.save(save_policy1)
tf_policy2_saver.save(save_policy2)


# Visualization

# # Run a game
# environment = eval_env
# policy = tf_agent.policy
# time_step = environment.reset()
#
# while not time_step.is_last():
#     print(time_step.observation)
#     action_step = policy.action(time_step)
#     time_step = environment.step(action_step.action)
# print('last:', time_step.observation)


# Plot
plt.bar(range(len(itrs)), itrs, width=0.4)
plt.show()
