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
    kTerminal
  };

  Phase phase_;

  int defender_position_;     // gdzie stoi defender
  int attack_target_;         // target wybrany przez attacker (-1 jeśli brak)
  int attack_remaining_;      // ile czasu do końca ataku
  int step_;                  // krok czasu
  int attacker_delay_;
  int defender_moves_;        // ile ruchów wykonał defender
};


struct SimpleGraph {
  std::vector<std::vector<int>> adj_matrix{
      {1, 1, 1},
      {1, 1, 1},
      {1, 1, 1},
  };

  std::vector<double> targets{1.0, 2.0, 3.0};
  std::vector<int> attack_duration{4, 2, 3};

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
  int MaxGameLength() const override { return 50; }
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

  // Used to implement the old observation API.
  std::shared_ptr<PatrolObserver> default_observer_;
  std::shared_ptr<PatrolObserver> info_state_observer_;
  std::shared_ptr<PatrolObserver> public_observer_;
  std::shared_ptr<PatrolObserver> private_observer_;

 const SimpleGraph& GetGraph() const { return graph_; }
 private:
  friend class PatrolState;
  // Number of players.
  int num_players_;
  int num_delays_;
  SimpleGraph graph_;

};


}  // namespace patrol
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PATROL_H_
