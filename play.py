from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import random
import collections
import os
import tempfile

from environment import FourInARow

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import random_tf_policy, policy_saver

tf.compat.v1.enable_v2_behavior()

policy1 = tf.compat.v2.saved_model.load("temp1")
policy2 = tf.compat.v2.saved_model.load("temp2")
env1 = FourInARow(policy1)
env2 = FourInARow(policy2)
env1 = tf_py_environment.TFPyEnvironment(env1)
env2 = tf_py_environment.TFPyEnvironment(env2)


def game(env):
    time_step = env.reset()

    while not time_step.is_last():
        print(time_step.observation)
        action = int(input("col: "))
        # action = policy.action(time_step).action
        time_step = env.step(action)
    print('last:', time_step.observation)
    if time_step.reward == 1:
        print("You won!")
    elif time_step.reward == 0.5:
        print("Tie")
    elif time_step.reward == -1:
        print("You lost!")
    else:
        print("In valid move")


game(env1)
print("new")
game(env2)
