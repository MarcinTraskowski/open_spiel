// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Include guard – prevents this header from being included multiple times
#ifndef OPEN_SPIEL_GAMES_PATROL_H_
#define OPEN_SPIEL_GAMES_PATROL_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace patrol {



class PatrolGame;
class PatrolObserver;

class PatrolState : public State {
 public:
  explicit PatrolState(std::shared_ptr<const Game> game);
  PatrolState(const PatrolState&) = default;

  Player CurrentPlayer() const override;

  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;

  std::unique_ptr<State> Clone() const override;

  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;

  /////// Maybe unnecessary
  std::string ActionToString(Player player, Action move) const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  //////


 protected:
  void DoApplyAction(Action move) override;

 private:
  friend class PatrolObserver;

  enum Phase {
    kChance,
    kDefender,
    kAttacker,
    kCaptureChance,
    kTerminal
  };

  Phase phase_;

  int defender_position_;   // current position of the defender
  int attack_target_;       // target chosen by the attacker (-1 if none)
  int attack_remaining_;    // remaining time until the attack completes
  int step_;                // current time step
  int attacker_delay_;      // delay before attacker starts the attack
  int defender_moves_;      // number of moves made by the defender
  std::vector<int> defender_history_; // history of defender positions
  bool defender_captured_;  // whether the defender has captured the attacker
};

struct SimpleGraph {
  std::vector<std::vector<int>> adj_matrix;
  std::vector<double> targets;
  std::vector<int> attack_duration;
  std::vector<std::vector<double>> coverage_matrix;
};


class PatrolGame : public Game {
 public:
  explicit PatrolGame(const GameParameters& params);
  std::unique_ptr<State> NewInitialState() const override;
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  absl::optional<double> UtilitySum() const override { return 0; }
  int MaxGameLength() const override;
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  int NumPlayers() const override { return num_players_; }

int NumDistinctActions() const override {
  return graph_.targets.size() * num_delays_;
}

int MaxChanceOutcomes() const override {
  return graph_.targets.size() * num_delays_; // num_nodes * num_delays
}

  // Used to implement the  observation API.
  std::shared_ptr<PatrolObserver> default_observer_;
  std::shared_ptr<PatrolObserver> info_state_observer_;
  std::shared_ptr<PatrolObserver> public_observer_;
  std::shared_ptr<PatrolObserver> private_observer_;

 const SimpleGraph& GetGraph() const { return graph_; }

 int GetNumDelays() const { return num_delays_; }

 int GetAttackerHistoryLength() const {
   return attacker_history_length_;
 }

 private:
  friend class PatrolState;
  friend class PatrolObserver;
  // Number of players.
  int num_players_;
  int num_delays_;
  SimpleGraph graph_;
  int attacker_history_length_; // how much of the history is revealed to the attacker



};


}  // namespace patrol
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PATROL_H_
