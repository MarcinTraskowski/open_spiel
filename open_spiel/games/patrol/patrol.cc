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

#include "open_spiel/games/patrol/patrol.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace patrol {
namespace {

// Default parameters.
constexpr int kDefaultPlayers = 2;

// Facts about the game
const GameType kGameType{/*short_name=*/"patrol",
                         /*long_name=*/"Patrol",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kDefaultPlayers)},
                          {"num_delays", GameParameter(3)}},
                         /*default_loadable=*/true,
                         /*provides_factored_observation_string=*/true,
                        };

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new PatrolGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

open_spiel::RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

/// PatrolObserver class, used to implement the new observation API
class PatrolObserver : public Observer {
 public:
  PatrolObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State& observed_state, int player,
                  Allocator* allocator) const override {
    const PatrolState& state =
        open_spiel::down_cast<const PatrolState&>(observed_state);

    // Minimal: encode just phase and position
    {
      auto out = allocator->Get("phase", {4});
      out.at(static_cast<int>(state.phase_)) = 1;
    }

    {
      auto out = allocator->Get("position", {15}); // max nodes for now, temporary
      if (state.defender_position_ >= 0)
        out.at(state.defender_position_) = 1;
    }
  }

  std::string StringFrom(const State& observed_state,
                        int player) const override {
    const PatrolState& state =
        open_spiel::down_cast<const PatrolState&>(observed_state);

    const char* phase_str =
        state.phase_ == PatrolState::kChance   ? "chance" :
        state.phase_ == PatrolState::kDefender ? "defender" :
        state.phase_ == PatrolState::kAttacker ? "attacker" :
                                                "terminal";

    return absl::StrCat(
        "phase=", phase_str,
        " pos=", state.defender_position_,
        " target=", state.attack_target_,
        " remaining=", state.attack_remaining_,
        " delay=", state.attacker_delay_
    );
  }

 private:
  IIGObservationType iig_obs_type_;
};

// The state of the game.


PatrolState::PatrolState(std::shared_ptr<const Game> game)
    : State(game),
      phase_(kChance),
      defender_position_(-1),
      attack_target_(-1),
      attack_remaining_(-1),
      step_(0),
      attacker_delay_(-1),
      defender_moves_(0) {}

Player PatrolState::CurrentPlayer() const {
  if (phase_ == kTerminal) {
    return kTerminalPlayerId;
  }

  switch (phase_) {
    case kChance:
      return kChancePlayerId;
    case kDefender:
      return 0;
    case kAttacker:
      return 1;
    default:
      return kTerminalPlayerId;
  }
}

std::string PatrolState::ToString() const {
  const char* phase_str =
      phase_ == kChance   ? "chance" :
      phase_ == kDefender ? "defender" :
      phase_ == kAttacker ? "attacker" :
                            "terminal";

  return absl::StrCat(
      "phase=", phase_str,
      " pos=", defender_position_,
      " target=", attack_target_,
      " remaining=", attack_remaining_,
      " step=", step_,
      " delay=", attacker_delay_
  );
}

bool PatrolState::IsTerminal() const {
  return phase_ == kTerminal;
}


std::vector<double> PatrolState::Returns() const {

  if (!IsTerminal()) {
    return {0.0, 0.0};
  }

  SPIEL_CHECK_GE(attack_target_, 0);

  const SimpleGraph& graph =
      static_cast<const PatrolGame&>(*game_).GetGraph();

  double value = graph.targets[attack_target_];

  // przykład: jeśli attack_remaining_ == 0 → atak się udał
  if (attack_remaining_ == 0) {
    return {-value, value};  // defender, attacker
  } else {
    return {value, -value};
  }
}


std::unique_ptr<State> PatrolState::Clone() const {
  return std::unique_ptr<State>(new PatrolState(*this));
}


std::vector<std::pair<Action, double>> PatrolState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(phase_ == kChance);

  std::vector<std::pair<Action, double>> outcomes;

  const auto& graph = static_cast<const PatrolGame&>(*game_).GetGraph();

  const int num_nodes = graph.targets.size(); 


  const auto& game =
    static_cast<const PatrolGame&>(*game_);
  int num_delays = game.num_delays_; // after this many nodes, attacker will act


  const double p = 1.0 / (num_nodes * num_delays);

  for (int start = 0; start < num_nodes; ++start) {
    for (int delay = 0; delay < num_delays; ++delay) {
      int action = start * num_delays + delay;
      outcomes.push_back({action, p});
    }
  }

  return outcomes;
}

void PatrolState::DoApplyAction(Action move) {
  const SimpleGraph& graph =
      static_cast<const PatrolGame&>(*game_).GetGraph();

  const int num_nodes = graph.targets.size();


  const auto& game =
      static_cast<const PatrolGame&>(*game_);

  int num_delays = game.num_delays_;

  // --------------------
  // 1. CHANCE
  // --------------------
  if (phase_ == kChance) {
    defender_position_ = move / num_delays;
    attacker_delay_    = move % num_delays;

    defender_moves_ = 0;
    step_ = 0;

    phase_ = kDefender;
    return;
  }

  // --------------------
  // 2. DEFENDER MOVE
  // --------------------
  if (phase_ == kDefender && attack_target_ == -1) {
    int new_pos = move;

    // update time (edge weight)
    step_ += graph.adj_matrix[defender_position_][new_pos];

    defender_position_ = new_pos;
    defender_moves_++;

    // attacker enters after k moves
    if (defender_moves_ >= attacker_delay_) {
      phase_ = kAttacker;
    }

    return;
  }

  // --------------------
  // 3. ATTACKER CHOOSES TARGET
  // --------------------
  if (phase_ == kAttacker) {
    attack_target_ = move;
    attack_remaining_ = graph.attack_duration[move];

    // immediate check: defender already there
    if (defender_position_ == attack_target_) {
      phase_ = kTerminal;
      return;
    }

    // attacker starts attack → back to defender
    phase_ = kDefender;
    return;
  }

  // --------------------
  // 4. DEFENDER DURING ATTACK
  // --------------------
  if (attack_target_ != -1 && phase_ == kDefender) {
    int new_pos = move;

    step_ += graph.adj_matrix[defender_position_][new_pos];
    defender_position_ = new_pos;

    attack_remaining_--;

    if (defender_position_ == attack_target_) {
      phase_ = kTerminal;  // defender wins
      return;
    }

    if (attack_remaining_ == 0) {
      phase_ = kTerminal;  // attacker wins
      return;
    }

    return;
  }

  SpielFatalError("Unexpected state in DoApplyAction");
}

std::vector<Action> PatrolState::LegalActions() const {
  if (IsTerminal()) return {};

  const auto& graph =
      static_cast<const PatrolGame&>(*game_).GetGraph();

  const int num_nodes = graph.targets.size();

  const auto& game =
      static_cast<const PatrolGame&>(*game_);

  int num_delays = game.num_delays_;

  // --------------------
  // CHANCE
  // --------------------
  if (phase_ == kChance) {
    std::vector<Action> actions;

    for (int start = 0; start < num_nodes; ++start) {
      for (int delay = 0; delay < num_delays; ++delay) {
        actions.push_back(start * num_delays + delay);
      }
    }

    return actions;
  }

  // --------------------
  // DEFENDER
  // --------------------
  if (phase_ == kDefender) {
  
    SPIEL_CHECK_GE(defender_position_, 0);
    std::vector<Action> actions;

    for (int j = 0; j < num_nodes; ++j) {
      if (graph.adj_matrix[defender_position_][j] >= 1) {
        actions.push_back(j);
      }
    }

    return actions;
  }

  // --------------------
  // ATTACKER
  // --------------------
  if (phase_ == kAttacker) {
    std::vector<Action> actions;

    for (int node = 0; node < num_nodes; ++node) {
      actions.push_back(node);
    }

    return actions;
  }

  return {};
}

std::string PatrolState::ActionToString(Player player, Action move) const {
  const auto& game =
      static_cast<const PatrolGame&>(*game_);

  int num_delays = game.num_delays_;

  // --------------------
  // CHANCE
  // --------------------
  if (player == kChancePlayerId) {
    int start = move / num_delays;
    int delay = move % num_delays;
    return absl::StrCat("start=", start, ",delay=", delay);
  }

  // --------------------
  // DEFENDER
  // --------------------
  if (player == 0) {
    return absl::StrCat("move_to=", move);
  }

  // --------------------
  // ATTACKER
  // --------------------
  if (player == 1) {
    return absl::StrCat("attack=", move);
  }

  return "unknown";
}


// std::string PatrolState::InformationStateString(Player player) const {
//   const PatrolGame& game = open_spiel::down_cast<const PatrolGame&>(*game_);
//   return game.info_state_observer_->StringFrom(*this, player);
// }

std::string PatrolState::InformationStateString(Player player) const {
  SPIEL_CHECK_NE(player, kChancePlayerId);

  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());

  if (defender_position_ < 0) {
    return absl::StrCat("p=", player, "|init");
  }

  return absl::StrCat(
      "p=", player,
      "|pos=", defender_position_,
      " step=", step_
  );
}
std::string PatrolState::ObservationString(Player player) const {

  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());

  const PatrolGame& game = open_spiel::down_cast<const PatrolGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void PatrolState::InformationStateTensor(Player player,
                                          absl::Span<float> values) const {

  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());

  ContiguousAllocator allocator(values);
  const PatrolGame& game = open_spiel::down_cast<const PatrolGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void PatrolState::ObservationTensor(Player player,
                                  absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());

  ContiguousAllocator allocator(values);
  const PatrolGame& game = open_spiel::down_cast<const PatrolGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}




// PatrolGame implementation.

PatrolGame::PatrolGame(const GameParameters& params)
    : Game(kGameType, params), num_players_(ParameterValue<int>("players")) {
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);

  num_delays_ = ParameterValue<int>("num_delays");


  default_observer_ = std::make_shared<PatrolObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<PatrolObserver>(kInfoStateObsType);
  private_observer_ = std::make_shared<PatrolObserver>(
      IIGObservationType{/*public_info*/false,
                         /*perfect_recall*/false,
                         /*private_info*/PrivateInfoType::kSinglePlayer});
  public_observer_ = std::make_shared<PatrolObserver>(
      IIGObservationType{/*public_info*/true,
                         /*perfect_recall*/false,
                         /*private_info*/PrivateInfoType::kNone});
}

std::unique_ptr<State> PatrolGame::NewInitialState() const {
  return std::unique_ptr<State>(new PatrolState(shared_from_this()));
}

std::vector<int> PatrolGame::InformationStateTensorShape() const {
  return {100}; ////////////////////// TEMPORART
}

std::vector<int> PatrolGame::ObservationTensorShape() const {
  return {100}; ///////////////////////////// TEMPORART
}

double PatrolGame::MaxUtility() const {

  double max_val = 0;
  for (double t : graph_.targets) {
    max_val = std::max(max_val, t);
  }
  return max_val;
}

double PatrolGame::MinUtility() const {
  return -MaxUtility();
}

std::shared_ptr<Observer> PatrolGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (params.empty()) {
    return std::make_shared<PatrolObserver>(
        iig_obs_type.value_or(kDefaultObsType));
  } else {
    return MakeRegisteredObserver(iig_obs_type, params);
  }
}

}  // namespace patrol
}  // namespace open_spiel
