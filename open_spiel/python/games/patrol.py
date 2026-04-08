# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Patrol game implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import numpy as np

import pyspiel


_NUM_PLAYERS = 2

_GAME_TYPE = pyspiel.GameType(
    short_name="patrol",
    long_name="Python Patrol Game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=3,
    max_chance_outcomes=3,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=10) 


class SimpleGraph:
    def __init__(self):
      
        self.nodes = [0, 1, 2]

        self.adj_matrix = [
            [1, 1, 1],  
            [1, 1, 1],  
            [1, 1, 1],  
        ]
        self.targets = {
            0: 1.0,
            1: 1.0,
            2: 3.0,
        }

        self.attack_duration = {
            0: 3,
            1: 3,
            2: 3,
        }

    # -------------------------
    # API

    def get_neighbors(self, node):
        return [
            j for j, connected in enumerate(self.adj_matrix[node])
            if connected >= 1
        ]

    def get_target_value(self, node):
        return self.targets[node]

    def get_attack_duration(self, node):
        return self.attack_duration[node]


class PatrolGame(pyspiel.Game):
  """A Python version of patrol game."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    self.graph = SimpleGraph()

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return PatrolState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return PatrolObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class PatrolState(pyspiel.State):
  """A python version of the Kuhn poker state."""

  def __init__(self, game):
    super().__init__(game)

    self.graph = game.graph
    self.observation_length = 2  # HARDCODED

    self.phase = "chance"   # "chance" / "defender" / "attacker"
    self.position = None

    self._history = []
    self.step = 0

    # attack
    self.attack_target = None
    self.attack_remaining = 0

    # terminal
    self._game_over = False
    self.success = None

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    if self.phase == "chance":
      return pyspiel.PlayerId.CHANCE
    if self.phase == "defender":
      return 0   # player 0
    if self.phase == "attacker":
      return 1   # player 1

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    
    assert player == self.current_player()
    
    if self.phase == "chance":
      return self.graph.nodes

    if self.phase == "defender":
      return self.graph.get_neighbors(self.position)

    if self.phase == "attacker":
      return self.graph.nodes
    
    return []

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    p = 1.0 / len(self.graph.nodes)
    return [(n, p) for n in self.graph.nodes]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    
    # -------------------------
    # CHANCE
    # -------------------------
    if self.phase == "chance":
      self.position = action
      self._history = [action]
      self.step = 1
      self.phase = "defender"
      return
    
    # -------------------------
    # DEFENDER
    # -------------------------
    if self.phase == "defender":
      self.position = action
      self._history.append(action)
      self.step += 1

      # --- attack is ongoing ---
      if self.attack_target is not None:

        # caught
        if self.position == self.attack_target:
          self._game_over = True
          self.success = True
          return

        self.attack_remaining -= 1

        if self.attack_remaining <= 0:
          self._game_over = True
          self.success = False
          return

        return

      # ---  attack is not  ongoing ---
      if self.step > 2:   # observation_length = 2 # HARDCODED
        self.phase = "attacker"

      return

    # -------------------------
    # ATTACKER
    # -------------------------
    if self.phase == "attacker":
      self.attack_target = action
      self.attack_remaining = self.graph.get_attack_duration(action)
      self.phase = "defender"
      return

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    if not self._game_over:
      return [0.0, 0.0]

    value = self.graph.get_target_value(self.attack_target)

    if self.success:
      return [value, -value]   # defender won
    else:
      return [-value, value]   # attacker won

  def __str__(self):
    return (
        f"[{self.phase}] "
        f"pos={self.position}, "
        f"hist={self._history}, "
        f"attack={self.attack_target}, "
        f"t={self.attack_remaining}"
    )

  def _action_to_string(self, player, action):
      if player == pyspiel.PlayerId.CHANCE:
          p = "C"
      elif player == 0:
          p = "D"  # defender
      elif player == 1:
          p = "A"  # attacker
      else:
          p = "?"

      if self.phase == "chance":
          return f"{p}: start@{action}"
      elif self.phase == "defender":
          return f"{p}: move->{action}"
      elif self.phase == "attacker":
          return f"{p}: attack->{action}"
      return f"{p}: {action}"


class PatrolObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Build the single flat tensor.
    self.tensor = np.zeros(1, np.float32)  # placeholder

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    pass  

  def string_from(self, state, player):

    k = state.observation_length

    if state.phase == "defender":
      hist = state._history[-k:]
      return f"D|{tuple(hist)}"

    if state.phase == "attacker":
      hist = state._history[-k:] 
      return f"A|{tuple(hist)}"

    return "C"


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, PatrolGame)
