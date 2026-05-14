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
#include <fstream>
#include <nlohmann/json.hpp>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

using json = nlohmann::json;
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
                          {"num_delays", GameParameter(2)},
                          {"attacker_history_length", GameParameter(-1)},
                          {"graph_path", GameParameter(std::string("graphs/star.json"))}},
                         /*default_loadable=*/true,
                         /*provides_factored_observation_string=*/true,
                        };


// Register game so it can be created via LoadGame("patrol")
std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new PatrolGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);


open_spiel::RegisterSingleTensorObserver single_tensor(kGameType.short_name);

/////////////// LOAD GRAPH ///////////////

SimpleGraph LoadGraphFromJson(const std::string& path) {
  std::ifstream file(path);

  SPIEL_CHECK_TRUE(file.is_open());

  json j;
  file >> j;

  SimpleGraph graph;

  graph.adj_matrix =
      j["adj_matrix"].get<std::vector<std::vector<int>>>();

  graph.targets =
      j["targets"].get<std::vector<double>>();

  graph.attack_duration =
      j["attack_duration"].get<std::vector<int>>();

  graph.coverage_matrix =
      j["coverage_matrix"]
        .get<std::vector<std::vector<double>>>();

  return graph;
}

/////////////////////////////////////////////


}  // namespace



/////////////// CLASS PatrolObserver ///////////////
class PatrolObserver : public Observer {
 public:

  // Constructor: defines what kind of observation this observer produces
  PatrolObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),

        // Store observation configuration (public/private info, recall, etc.)
        // Used by OpenSpiel to control what each player can "see"
        iig_obs_type_(iig_obs_type) {}


  void WriteTensor(const State& observed_state, int player,
                  Allocator* allocator) const override {

    const PatrolState& state =
        open_spiel::down_cast<const PatrolState&>(observed_state);

    const auto& game =
        open_spiel::down_cast<const PatrolGame&>(*state.game_);

    int N = game.GetGraph().targets.size();

    // MAX GAME LENGTH
    int max_len = game.MaxGameLength();

    // --- player (one-hot size 2) ---
    {
      auto out = allocator->Get("player", {2});
      for (int i = 0; i < 2; ++i) {
        out.at(i) = 0.0f;
      }

      out.at(player) = 1.0f;
    }

    // --- full history (flatten T x N) ---
    {
      auto out = allocator->Get("history", {max_len * N});

      // Initialize with zeros
      int total_size = max_len * N;

      for (int i = 0; i < total_size; ++i) {
        out.at(i) = 0.0f;
      }
      const auto& hist = state.defender_history_;

      for (int t = 0; t < hist.size() && t < max_len; ++t) {
        int pos = hist[t];
        if (pos >= 0 && pos < N) {
          out.at(t * N + pos) = 1;
        }
      }
    }
  }


  ////////////////////// old version //////////////////
  // // Encode the state as a feature vector using one-hot encoding:
  // void WriteTensor(const State& observed_state, int player,
  //                 Allocator* allocator) const override {
    
  //   // Cast generic State → PatrolState (we know what game this is)
  //   const PatrolState& state =
  //       open_spiel::down_cast<const PatrolState&>(observed_state);

  //    // -------- encode phase (one-hot of size 4) --------
  //   {
  //     // Allocate a slice of the tensor named "phase" with size 4
  //     auto out = allocator->Get("phase", {4});
  //     out.at(static_cast<int>(state.phase_)) = 1;
  //   }

  //    // -------- encode defender position (one-hot of size N) --------
  //   {
  //     // Get access to the game object to read graph size
  //     const auto& game =
  //         open_spiel::down_cast<const PatrolGame&>(*state.game_);
      
  //     // Number of nodes in the graph
  //     int N = game.GetGraph().targets.size();
      
  //     auto out = allocator->Get("position", {N});
  //     if (state.defender_position_ >= 0) {
  //       out.at(state.defender_position_) = 1;
  //     }
  //   }
  // }
  ////////////////////// old version //////////////////


  std::string StringFrom(const State& observed_state,
                        int player) const override {
                    
    const PatrolState& state =
        open_spiel::down_cast<const PatrolState&>(observed_state);

    const auto& game =
        open_spiel::down_cast<const PatrolGame&>(*state.game_);

    const auto& hist = state.defender_history_;

    std::vector<int> visible_hist;

    // defender sees full history
    if (player == 0 ||
        game.attacker_history_length_ < 0 ||
        hist.size() <= game.attacker_history_length_) {

      visible_hist = hist;

    } else {

      visible_hist = std::vector<int>(
          hist.end() - game.attacker_history_length_,
          hist.end()
      );
    }


    return absl::StrCat(
        "p=", player,
        "|def_hist=",
        visible_hist.empty()
            ? "init"
            : absl::StrJoin(visible_hist, ",")
    );
  }

  // VERSION WITH LAST 2 MOVES ONLY (instead of full history)
  // std::string StringFrom(const State& observed_state,
  //                       int player) const override {
  //   const PatrolState& state =
  //       open_spiel::down_cast<const PatrolState&>(observed_state);

  //   // weź ostatnie 2 ruchy (albo mniej jeśli nie ma)
  //   std::string hist_str;
  //   int h_size = state.defender_history_.size();

  //   if (h_size == 0) {
  //     hist_str = "init";
  //   } else if (h_size == 1) {
  //     hist_str = absl::StrCat(state.defender_history_[0]);
  //   } else {
  //     hist_str = absl::StrCat(
  //         state.defender_history_[h_size - 2], ",",
  //         state.defender_history_[h_size - 1]);
  //   }

  //   return absl::StrCat(
  //       "p=", player,
  //       "|hist=", hist_str
  //   );
  // }

 private:
  // Specifies what information this observer exposes:
  // - whether observations include public info
  // - whether perfect recall is assumed
  // - what private information (if any) is visible to the player
  IIGObservationType iig_obs_type_;
};

/////////////////////////////////////////////


/////////////// CLASS PatrolState ///////////////


PatrolState::PatrolState(std::shared_ptr<const Game> game)
    : State(game),
      phase_(kChance),
      defender_position_(-1),
      attack_target_(-1),
      attack_remaining_(-1),
      step_(0),
      attacker_delay_(-1),
      defender_moves_(0),
      defender_captured_(false) {}


Player PatrolState::CurrentPlayer() const {
  if (phase_ == kTerminal) {
    return kTerminalPlayerId;
  }

  switch (phase_) {
    case kChance:
      return kChancePlayerId;
    case kCaptureChance:
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
      phase_ == kCaptureChance ? "capture_chance" :
      phase_ == kDefender ? "defender" :
      phase_ == kAttacker ? "attacker" :
                            "terminal";

  return absl::StrCat(
      "phase=", phase_str,
      " defender_position_=", defender_position_,
      " attack_target_=", attack_target_,
      " attack_remaining_=", attack_remaining_,
      " step_=", step_,
      " attacker_delay_=", attacker_delay_
  );
}


bool PatrolState::IsTerminal() const {
  return phase_ == kTerminal;
}


std::vector<double> PatrolState::Returns() const {

  if (!IsTerminal()) {
    return {0.0, 0.0};
  }

  // checking >=
  SPIEL_CHECK_GE(attack_target_, 0);

  const SimpleGraph& graph =
      static_cast<const PatrolGame&>(*game_).GetGraph();

  double value = graph.targets[attack_target_];

  if (defender_captured_) {
    return {value, -value};  // defender, attacker
  } else  {
    return {-value, value};  // defender, attacker
  }
}


std::unique_ptr<State> PatrolState::Clone() const {
  return std::unique_ptr<State>(new PatrolState(*this));
}


std::vector<std::pair<Action, double>> PatrolState::ChanceOutcomes() const {

  SPIEL_CHECK_TRUE(phase_ == kChance || phase_ == kCaptureChance);

  const auto& graph = static_cast<const PatrolGame&>(*game_).GetGraph();

  if (phase_ == kCaptureChance) {

    double p =
        graph.coverage_matrix
            [defender_position_]
            [attack_target_];

    return {
        {0, p},          // capture success
        {1, 1.0 - p}     // capture failure
    };
  }

  // --------------------
  // INITIAL CHANCE
  // 

  std::vector<std::pair<Action, double>> outcomes;

  const int num_nodes = graph.targets.size(); 

  const auto& game =
    static_cast<const PatrolGame&>(*game_);

  int num_delays = game.num_delays_; // after this many nodes, attacker will act

  // Sample initial defender position and attacker delay.
  // Each (position, delay) pair is assigned equal probability.
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

    defender_history_.clear();
    defender_history_.push_back(defender_position_);

    // checking defender_moves_ >= attacker_delay_
    if (attacker_delay_ == 0) {
      phase_ = kAttacker;
    }
    else {
    phase_ = kDefender;
    }
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
    defender_history_.push_back(new_pos);

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

    // if (attack_target_ == defender_position_) {
    //   phase_ = kTerminal;  // defender wins immediately
    //   return;
    // }

    // attacker starts attack → back to defender
    phase_ = kDefender;
    return;
  }

  // --------------------
  // 4. DEFENDER DURING ATTACK
  // --------------------
  if (attack_target_ != -1 && phase_ == kDefender) {
    int new_pos = move;
    int travel_time = graph.adj_matrix[defender_position_][new_pos];
  

    step_ += travel_time;
    attack_remaining_ -= travel_time;

    if (attack_remaining_ < 0) {
      phase_ = kTerminal;  // attacker wins
      return;
    }

    // update defender position and history
    defender_position_ = new_pos;
    defender_history_.push_back(new_pos);

    phase_ = kCaptureChance;

    return;
  }

  // --------------------
  // 5. CAPTURE CHANCE
  // --------------------

  if (phase_ == kCaptureChance) {

    if (move == 0) {
      defender_captured_ = true;
      phase_ = kTerminal;
    } else {
      defender_captured_ = false;
      phase_ = kDefender;
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
  // CHANCE START
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
  // CHANCE Capture
  // --------------------
  if (phase_ == kCaptureChance) {
    return {0, 1};
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
    if (phase_ == kChance) {
      int start = move / num_delays;
      int delay = move % num_delays;
      return absl::StrCat("start=", start, ",delay=", delay);
    }
    if (phase_ == kCaptureChance) {

      if (move == 0) {
        return "capture_success";
      } else {
        return "capture_failure";
      }
    }
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


std::string PatrolState::InformationStateString(Player player) const {
  const PatrolGame& game = open_spiel::down_cast<const PatrolGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

//// Version without observer

// std::string PatrolState::InformationStateString(Player player) const {

//   // Information state is defined only for actual players (not chance)
//   SPIEL_CHECK_NE(player, kChancePlayerId);

//   // Ensure valid player index (0 <= player < num_players)
//   SPIEL_CHECK_GE(player, 0);
//   SPIEL_CHECK_LT(player, game_->NumPlayers());

//   if (defender_position_ < 0) {
//     return absl::StrCat("p=", player, "|init");
//   }

//   return absl::StrCat(
//       "p=", player,
//       "|def_hist=", absl::StrJoin(defender_history_, ",")
//   );
// }

////

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

/////////////////////////////////////////////


/////////////// CLASS PatrolGame ///////////////

PatrolGame::PatrolGame(const GameParameters& params)
    : Game(kGameType, params), num_players_(ParameterValue<int>("players")) {

  // Check that number of players is bigger than min and smaller than max.
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);

  num_delays_ = ParameterValue<int>("num_delays");

  attacker_history_length_ =
      ParameterValue<int>("attacker_history_length");

  std::string graph_path =
      ParameterValue<std::string>("graph_path");

  graph_ = LoadGraphFromJson(graph_path);

  ///////////// FILL GRAPH DEFINITION /////////////

  //  //GRAPH 1
  // graph_.adj_matrix = {{1,1,1},{1,1,1},{1,1,1}};
  // graph_.targets = {1.0, 1.0, 1.0};
  // graph_.attack_duration = {2,2,2};

  //  //GRAPH GDYNIA (2)
  // graph_.adj_matrix = {
  //     {0,1,1,1,0,0,0,0,0,0},
  //     {1,0,1,1,0,0,0,0,0,0},
  //     {1,1,0,1,0,0,0,0,0,0},
  //     {1,1,1,0,1,0,0,0,0,1},
  //     {0,0,0,1,0,0,0,0,1,1},
  //     {0,0,0,0,0,0,1,1,0,0},
  //     {0,0,0,0,0,1,0,1,0,0},
  //     {0,0,0,0,0,1,1,0,1,0},
  //     {0,0,0,0,1,0,0,1,0,1},
  //     {0,0,0,1,1,0,0,0,1,0}
  // };

  // graph_.targets = {
  //     0.0, 0.0, 1.0, 0.0, 1.0,
  //     1.0, 1.0, 1.0, 1.0, 1.0
  // };

  // graph_.attack_duration = {
  //     3,3,3,3,3,3,3,3,3,3
  // };

  // JOHN ET AL. GRAPH (8 nodes)

  // graph_.adj_matrix = {
  //     {1,3,3,5,4,6,3,5},
  //     {3,1,5,4,2,4,4,5},
  //     {3,5,1,7,6,8,3,4},
  //     {6,4,7,1,5,6,4,7},
  //     {4,3,6,5,1,3,5,5},
  //     {6,4,8,5,3,1,6,7},
  //     {2,5,3,5,6,7,1,5},
  //     {3,5,2,7,6,7,3,1}
  // };

  // graph_.targets = {
  //     1.0, 1.0, 1.0, 1.0,
  //     1.0, 1.0, 1.0, 1.0
  // };

  // graph_.attack_duration = {
  //     8, 6, 11, 10,
  //     6, 10, 9, 10
  // };

  // // JOHN ET AL. GRAPH (12 nodes)

  // graph_.adj_matrix = {
  //     {1,3,3,5,4,6,3,5,7,4,6,6},
  //     {3,1,5,4,2,4,4,5,5,3,5,5},
  //     {3,5,1,7,6,8,3,4,9,4,8,7},
  //     {6,4,7,1,5,6,4,7,5,6,6,7},
  //     {4,3,6,5,1,3,5,5,6,3,4,4},
  //     {6,4,8,5,3,1,6,7,3,6,2,3},
  //     {2,5,3,5,6,7,1,5,7,5,7,8},
  //     {3,5,2,7,6,7,3,1,9,3,7,5},
  //     {8,6,9,4,6,4,6,9,1,8,5,7},
  //     {4,3,4,6,3,5,5,3,7,1,5,3},
  //     {6,4,8,6,4,2,6,6,4,5,1,3},
  //     {6,4,6,6,3,3,6,4,5,3,2,1}
  // };

  // graph_.targets = {
  //     1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
  //     1.0, 1.0, 1.0, 1.0, 1.0, 1.0
  // };

  // graph_.attack_duration = {
  //     8, 6, 11, 10,
  //     6, 10, 9, 10,
  //     11, 9, 10, 8
  // };

  // star graph from shield
  // graph_.adj_matrix = {
  //     {1,1,1,1},
  //     {1,1,0,0},
  //     {1,0,1,0},
  //     {1,0,0,1}
  // };
  // graph_.targets = {
  //     0.0, 1.0, 1.0, 1.0
  // };

  // graph_.attack_duration = {
  //     2,2,2,2
  // };


  // Default observation type for imperfect-information games.
  // Used by ObservationTensor / ObservationString.
  // Typically contains only public information (visible to all players)
  // and does NOT include full history (no perfect recall).
  default_observer_ = std::make_shared<PatrolObserver>(kDefaultObsType);

  // Information state observation type.
  // Used by InformationStateTensor / InformationStateString.
  // Represents the player's information set:
  // - includes all information available to the player (public + private)
  // - typically assumes perfect recall (full history of observations/actions)
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
  int N = graph_.targets.size();
  int max_len = MaxGameLength();

  return {2 + max_len * N};
}

std::vector<int> PatrolGame::ObservationTensorShape() const {
  int N = graph_.targets.size();
  int max_len = MaxGameLength();

  return {2 + max_len * N};
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

int PatrolGame::MaxGameLength() const {
  int max_attack = 0;
  for (int d : graph_.attack_duration) {
    max_attack = std::max(max_attack, d);
  }

  // delay  + defender + capture_chance + small buffer
  return num_delays_ + 2 * max_attack + 10; // small buffer
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

/////////////////////////////////////////////

}  // namespace patrol
}  // namespace open_spiel
