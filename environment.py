from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import random

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


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


def check_win(grid, col, row, turn, win_amount=4):
    return (check_win_in_row(grid[row, :], turn, win_amount) or check_win_in_row(grid[:, col], turn, win_amount)
            or check_win_in_row(np.diagonal(grid, offset=col-row), turn, win_amount)
            or check_win_in_row(np.diagonal(np.fliplr(grid), offset=7-1-(col+row)), turn, win_amount))


class FourInARow(py_environment.PyEnvironment):

    def __init__(self):
        self.IDENTIFIER = 1
        self.LENGTH = 6
        self.WIDTH = 7
        self._action_spec = \
            array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=self.LENGTH, name='action')  # 7
        self._observation_spec = \
            array_spec.BoundedArraySpec(shape=(self.LENGTH, self.WIDTH), dtype=np.int32, minimum=0, maximum=2, name='observation')
        self._state = self.crete_board_arr()
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self.crete_board_arr()
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def is_valid_move(self, action: int):
        return True if self._state[0][action] == 0 else False

    def random_move(self):
        """
        Choose a random move and update the state
        :return: The num col of the move and the row
        """
        cols = []
        for col in range(self.WIDTH):
            if self.is_valid_move(col):
                cols.append(col)
        col = random.choice(cols)
        place_row = None
        for row in range(1, 7):
            if self._state[-row][col] == 0:
                place_row = row
                break
        if place_row is None:
            raise Exception('function random move dose not work')

        self._state[-place_row][col] = 2
        return col, place_row

    def _step(self, action: int):
        # player move is always 1, identifier

        # TODO clean the code, too many similar blokes of code, too long function

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Find the place
        place_row = None
        for row in range(1, 7):
            if self._state[-row][action] == 0:
                place_row = row
                break

        # Make sure it is a valid move
        if place_row is None:
            self._episode_ended = True
            reward = -10
            return ts.termination(np.array(self._state, dtype=np.int32), reward)

        # Update the board
        elif 0 <= action < 7:
            self._state[-place_row][action] = self.IDENTIFIER
        # Invalid move
        else:
            raise ValueError('`action` should be betttwn 0 and 7.')

        # I won
        if check_win(self._state, action, 6 - place_row, self.IDENTIFIER):
            self._episode_ended = True
            reward = 1
            return ts.termination(np.array(self._state, dtype=np.int32), reward)

        # Check if the board is full
        is_full = self.board_is_full()
        if is_full:
            self._episode_ended = True
            return ts.transition(np.array(self._state, dtype=np.int32), reward=0.5)

        # Make a random move
        col, row = self.random_move()

        # He won
        if check_win(self._state, col, 6 - row, 2):
            self._episode_ended = True
            reward = -1
            return ts.termination(np.array(self._state, dtype=np.int32), reward)

        # Check if the board is full
        is_full = self.board_is_full()
        if is_full:
            self._episode_ended = True
            return ts.transition(np.array(self._state, dtype=np.int32), reward=0.5)

        return ts.transition(
            np.array(self._state, dtype=np.int32), reward=0.01, discount=0.99)

    def board_is_full(self):
        for col in range(7):
            if self._state[0][col] == 0:
                return False
        return True

    def crete_board_arr(self):
        return np.zeros((6, 7), dtype=int)


env = FourInARow()
# utils.validate_py_environment(env, episodes=10)   # env test

# tf_env = tf_py_environment.TFPyEnvironment(env)
# print(isinstance(tf_env, tf_environment.TFEnvironment))
# print("TimeStep Specs:", tf_env.time_step_spec())
# print("Action Specs:", tf_env.action_spec())


def run():
    """
    Simulate a game
    :return: None
    """
    time_step = env.reset()
    print(time_step.observation)
    cumulative_reward = time_step.reward

    for _ in range(8):
      time_step = env.step(3)
      print(time_step.observation)
      cumulative_reward += time_step.reward

    time_step = env.step(3)
    print(time_step.observation)
    cumulative_reward += time_step.reward
    print('Final Reward = ', cumulative_reward)

