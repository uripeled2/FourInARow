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

tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())


def check_win_in_row(line, turn, win_amount=4):
    count = 0
    for c in range(len(line)):
        if line[c] == turn:
            count += 1
            if count >= win_amount:
                return True
        else:
            count = 0
    return False


class FourInARow(py_environment.PyEnvironment):

    def __init__(self, agent2_policy: callable, rest: bool = True):
        self.IDENTIFIER = 1
        self.OPPOSITE_IDENTIFIER = 2
        self.LENGTH = 6
        self.WIDTH = 7
        self._action_spec = \
            array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=self.LENGTH, name='action')  # 7
        self._observation_spec = \
            array_spec.BoundedArraySpec(shape=(self.LENGTH, self.WIDTH), dtype=np.int32, minimum=0, maximum=2, name='observation')
        self.agent2_policy = agent2_policy
        if rest:
            self._reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state1, self._state2 = self.crete_boards()
        self._episode_ended = False
        return ts.restart(np.array(self._state1, dtype=np.int32))

    def _tf_time_step(self, state = None):
        if state is not None:
            tf_state = tf.convert_to_tensor([np.array(state, dtype=np.int32)])
        else:
            tf_state = tf.convert_to_tensor([np.array(self._state2, dtype=np.int32)])
        tf_reward = tf.convert_to_tensor([np.array(0.0, dtype=np.float32)], dtype=tf.float32)
        agent2_time_step = ts.termination(tf_state, tf_reward)
        return agent2_time_step

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        row = self.find_row(action)
        # Make sure it is a valid move
        if row is None:
            self._episode_ended = True
            return ts.termination(np.array(self._state1, dtype=np.int32), reward=-10)

        # Update the boards
        win, full = self.update(self.IDENTIFIER, self.OPPOSITE_IDENTIFIER, action, row)
        if win:
            return ts.termination(np.array(self._state1, dtype=np.int32), reward=1)
        elif full:
            return ts.termination(np.array(self._state1, dtype=np.int32), reward=0.5)

        # Make a second action
        agent2_time_step = self._tf_time_step()

        if self.agent2_policy is None:
            raise Exception('Error!, agent2 dose not exists')

        action = self.agent2_policy.action(agent2_time_step).action

        # Update the boards
        win, full = self.update(self.OPPOSITE_IDENTIFIER, self.IDENTIFIER, action)
        if win:
            return ts.termination(np.array(self._state1, dtype=np.int32), reward=-1)
        elif full:
            return ts.termination(np.array(self._state1, dtype=np.int32), reward=0.5)

        time_step = ts.transition(np.array(self._state1, dtype=np.int32), reward=0.01, discount=1.0)
        # print("temp:", temp_time_step)
        # print("tf_temp:", tf_temp_time_step)
        # print("real:", time_step)
        return time_step

    def update(self, ide: int, ops: int, col: int, row: int = None):
        """
        Update both start1, state2 and _episode_ended

        :param ide: identifier as agent1 see it
        :param ops: identifier as agent2 see it
        :param col:
        :param row:
        :return: bool, bool. If there is a winerr and if the board is full
        """
        row = self.find_row(col) if row is None else row
        if row is None:
            col, row = self.random_move()
        self._state1[-row][col] = ide
        self._state2[-row][col] = ops

        if self.check_win(col, self.LENGTH - row, ide):
            self._episode_ended = True
            return True, False
        if self.board_is_full():
            self._episode_ended = True
            return False, True
        return False, False

    def find_row(self, col: int):
        place_row = None
        for row in range(1, self.LENGTH + 1):
            if self._state1[-row][col] == 0:
                place_row = row
                break
        return place_row

    def random_move(self):
        """
        Choose a random valid move
        :return: The num col of the move and the row
        """
        cols = []
        for col in range(self.WIDTH):
            if self.is_valid_move(col):
                cols.append(col)
        col = random.choice(cols)
        row = self.find_row(col)
        return col, row

    def is_valid_move(self, action: int):
        return self._state1[0][action] == 0

    def board_is_full(self):
        for col in range(7):
            if self._state1[0][col] == 0:
                return False
        return True

    def crete_boards(self):
        # Crete empty boards
        board1 = np.zeros((self.LENGTH, self.WIDTH), dtype=int)  # [[0 * 7] * 6]
        board2 = np.zeros((self.LENGTH, self.WIDTH), dtype=int)  # [[0 * 7] * 6]

        # What if the other player start
        if random.random() > 0.5:
            time_step = self._tf_time_step(state=board2)
            action = self.agent2_policy.action(time_step).action
            board1[-1][action] = self.OPPOSITE_IDENTIFIER
            board2[-1][action] = self.IDENTIFIER

        return board1, board2

    def check_win(self, col, row, turn, win_amount=4):
        return (check_win_in_row(self._state1[row, :], turn, win_amount)
                or check_win_in_row(self._state1[:, col], turn, win_amount)
                or check_win_in_row(np.diagonal(self._state1, offset=col - row), turn, win_amount)
                or check_win_in_row(np.diagonal(np.fliplr(self._state1), offset=7 - 1 - (col + row)), turn, win_amount))


# Tests

# policy_dir = os.path.join(tempdir, 'policy')
# my_policy = tf.compat.v2.saved_model.load(policy_dir)
# # setup env
# env = FourInARow(my_policy)
# utils.validate_py_environment(env, episodes=10)  # Pass
# env = tf_py_environment.TFPyEnvironment(env)
#
# random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())

# # Save
# tf_policy_saver = policy_saver.PolicySaver(random_policy)
# tf_policy_saver.save("random_po")
# policy_dir = os.path.join(tempdir, 'policy')
# tf_policy_saver = policy_saver.PolicySaver(random_policy)
# tf_policy_saver.save(policy_dir)


def run(times: int = 8):
    """
    Simulate a game
    :return: None
    """
    time_step = env.reset()
    print(time_step.observation)
    cumulative_reward = time_step.reward

    for _ in range(times):
        act = random_policy.action(time_step).action
        act = 3
        time_step = env.step(act)
        print(time_step.observation)
        cumulative_reward += time_step.reward

    time_step = env.step(3)
    print(time_step.observation)
    cumulative_reward += time_step.reward
    print('Final Reward = ', cumulative_reward)


# run(times=20)


