# --- START OF FILE organized_poker_bot/cfr/cfr_trainer.py ---
"""
Implementation of Counterfactual Regret Minimization (CFR) for poker.
Utilizes External Sampling style updates and Linear CFR weighting.
(Refactored V19: Use shared info_set_util.py for key generation)
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict
import sys
import traceback
import time

# Imports (Absolute)
try:
    from organized_poker_bot.cfr.information_set import InformationSet
    # Card/Action Abstraction might only be needed by the utility function now
    from organized_poker_bot.cfr.action_abstraction import ActionAbstraction # Keep if used below
    from organized_poker_bot.game_engine.game_state import GameState
    # Import the NEW Utility for key generation
    from organized_poker_bot.cfr.info_set_util import generate_info_set_key
    # Keep Card import if game_state factory or other parts need it type hints
    from organized_poker_bot.game_engine.card import Card
except ImportError as e:
    print(f"FATAL Import Error in cfr_trainer.py: {e}")
    print("Ensure 'organized_poker_bot' is in PYTHONPATH or run from root.")
    sys.exit(1)


# Recursion limit setup
try:
    current_limit = sys.getrecursionlimit()
    target_limit = 3000
    if current_limit < target_limit:
        sys.setrecursionlimit(target_limit)
        current_limit = sys.getrecursionlimit()
    CFRTrainer_REC_LIMIT = max(500, current_limit - 100)
except Exception as e:
    CFRTrainer_REC_LIMIT = 1000
    print(f"WARN: Failed to adjust recursion limit: {e}. Using default: {CFRTrainer_REC_LIMIT}")


class CFRTrainer:
    RECURSION_DEPTH_LIMIT = CFRTrainer_REC_LIMIT

    def __init__(self, game_state_class, num_players=2,
                 use_action_abstraction=True, use_card_abstraction=True, # These flags might now be obsolete if handled solely by info_set_util
                 custom_get_actions_func=None):
        if not callable(game_state_class):
            raise TypeError("GS class !callable")
        self.game_state_class = game_state_class
        self.num_players = num_players
        self.information_sets = {}
        self.iterations = 0
        # Note: The 'use_*_abstraction' flags might be conceptually moved to info_set_util config,
        # but we keep them here for potential trainer-specific logic if ever needed.
        # Ensure consistency between these flags and info_set_util's configuration.
        self.use_action_abstraction = use_action_abstraction
        self.use_card_abstraction = use_card_abstraction
        self.training_start_time = None
        self.get_actions_override = custom_get_actions_func
        # Disable standard action abstraction if a custom one is provided
        if custom_get_actions_func and self.use_action_abstraction:
            self.use_action_abstraction = False


    def train(self, iterations=1000, checkpoint_freq=100, output_dir=None, verbose=False,
              log_freq_override=None): # Added log_freq_override parameter
        """ Trains the CFR model using External Sampling and Linear Weighting. """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if self.training_start_time is None:
            self.training_start_time = time.time()

        start_iter = self.iterations
        end_iter = self.iterations + iterations
        num_iterations_this_run = iterations

        print(f"Starting Linear CFR Training: Iterations {start_iter + 1} to {end_iter}...")

        pbar_disable = (num_iterations_this_run <= 100 and not verbose)
        pbar = tqdm(range(start_iter, end_iter), desc="CFR Training", initial=start_iter, total=end_iter, disable=pbar_disable, unit="iter")

        total_train_time_sec_this_run = 0.0
        log_print_frequency = log_freq_override if log_freq_override else max(1, min(1000, num_iterations_this_run // 20))
        if num_iterations_this_run > log_print_frequency :
            print(f"(Logging progress approx every {log_print_frequency} iterations)")

        for i in pbar: # Main Training Loop
            iter_start_time = time.time()
            current_iter_num = i + 1
            if not pbar_disable:
                pbar.set_description(f"CFR Iter {current_iter_num}")

            game_state = None
            initial_stacks_hand = []
            try: # Get Initial Stacks
                temp_gs = self.game_state_class(self.num_players)
                default_stack = getattr(temp_gs, 'starting_stack', 10000.0)
                initial_stacks_hand = getattr(temp_gs, 'player_stacks', [float(default_stack)] * self.num_players)[:]
                if not initial_stacks_hand:
                    initial_stacks_hand = [float(default_stack)] * self.num_players
            except Exception:
                initial_stacks_hand = [10000.0] * self.num_players

            try: # Setup Hand
                game_state = self.game_state_class(self.num_players)
                stacks_for_hand = initial_stacks_hand[:]
                dealer_pos = current_iter_num % self.num_players
                game_state.start_new_hand(dealer_pos=dealer_pos, player_stacks=stacks_for_hand)
                if game_state.is_terminal() or game_state.current_player_idx == -1:
                    continue
            except Exception as e:
                print(f"ERROR starting hand for Iter {current_iter_num}: {e}")
                traceback.print_exc()
                continue

            # External Sampling Loop
            reach_probs = np.ones(self.num_players, dtype=float)
            iter_utilities_perspectives = []
            failed_perspectives_count = 0
            for p_idx in range(self.num_players):
                perspective_utility = 0.0
                try:
                    perspective_utility = self._calculate_cfr(
                        game_state.clone(),
                        reach_probs.copy(),
                        p_idx,
                        initial_stacks_hand[:],
                        float(current_iter_num),
                        0.0, 0, verbose
                    )
                    iter_utilities_perspectives.append(perspective_utility)
                except RecursionError as re:
                    print(f"\nFATAL: Rec Limit P{p_idx} Iter {current_iter_num}.")
                    pbar.close()
                    final_strat_on_error = self.get_strategy()
                    if output_dir:
                        self._save_final_strategy(output_dir, final_strat_on_error)
                    raise re
                except Exception as e:
                    print(f"ERROR CFR calc P{p_idx} Iter {current_iter_num}: {e}")
                    traceback.print_exc()
                    iter_utilities_perspectives.append(None)
                    failed_perspectives_count += 1

            # Iteration Update & Logging
            iter_duration_sec = time.time() - iter_start_time
            total_train_time_sec_this_run += iter_duration_sec
            if failed_perspectives_count < self.num_players:
                self.iterations = current_iter_num
                if output_dir and (self.iterations % checkpoint_freq == 0):
                    self._save_checkpoint(output_dir, self.iterations)

                valid_utils = [u for u in iter_utilities_perspectives if isinstance(u, (int,float)) and not np.isnan(u) and not np.isinf(u)]
                avg_util_iter = f"{np.mean(valid_utils):.3f}" if valid_utils else "N/A"

                if not pbar_disable:
                    pbar.set_postfix({"Sets": len(self.information_sets), "AvgUtil": avg_util_iter, "LastT": f"{iter_duration_sec:.2f}s"}, refresh=True)

                if self.iterations % log_print_frequency == 0 or self.iterations == end_iter:
                    time_elapsed = time.time() - self.training_start_time
                    avg_iter_time = time_elapsed / self.iterations if self.iterations > 0 else 0
                    checkpoint_saved_msg = "> CHK" if output_dir and (self.iterations % checkpoint_freq == 0) else ""
                    print(f"   Iter {self.iterations}/{end_iter} | InfoSets: {len(self.information_sets):,} | AvgUtil: {avg_util_iter} | AvgTime: {avg_iter_time:.3f}s {checkpoint_saved_msg}")
            else:
                print(f"WARN: Skipping iter {current_iter_num} update - all perspectives failed.")

        # End Training Loop
        pbar.close()
        avg_time_per_iter_run = total_train_time_sec_this_run / num_iterations_this_run if num_iterations_this_run > 0 else 0
        total_elapsed_time = time.time() - self.training_start_time
        print(f"\nTraining loop finished ({num_iterations_this_run} iter). AvgIterT={avg_time_per_iter_run:.4f}s. TotalT={total_elapsed_time:.2f}s")
        final_strat = self.get_strategy()
        if output_dir:
            self._save_final_strategy(output_dir, final_strat)
        return final_strat


    def _calculate_cfr(self, game_state, reach_probs, player_idx, initial_stacks, weight, prune_threshold, depth, verbose):
        """ Recursive CFR function using shared key generation (V19) """

        # Base Cases
        if game_state.is_terminal():
            utility = 0.0
            try:
                utility_val = game_state.get_utility(player_idx, initial_stacks)
                utility = float(utility_val) if isinstance(utility_val, (int, float)) and not np.isnan(utility_val) and not np.isinf(utility_val) else 0.0
            except Exception:
                pass
            return utility
        if depth > self.RECURSION_DEPTH_LIMIT:
            return 0.0

        # Handle Inactive Player
        acting_player_idx = game_state.current_player_idx
        if not (0 <= acting_player_idx < self.num_players):
            return 0.0

        is_folded = game_state.player_folded[acting_player_idx] if acting_player_idx < len(game_state.player_folded) else True
        is_all_in = game_state.player_all_in[acting_player_idx] if acting_player_idx < len(game_state.player_all_in) else True

        if is_folded or is_all_in:
            temp_state = game_state.clone()
            original_turn_idx = temp_state.current_player_idx
            temp_state.rotate_turn()
            if temp_state.current_player_idx == original_turn_idx or temp_state.is_terminal():
                utility = 0.0
                try:
                    utility_val = temp_state.get_utility(player_idx, initial_stacks)
                    utility = float(utility_val) if isinstance(utility_val, (int,float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
                except Exception:
                    pass
                return utility
            else:
                return self._calculate_cfr(temp_state, reach_probs, player_idx, initial_stacks, weight, prune_threshold, depth + 1, verbose)

        # --- Active player's turn ---
        # <<<--- Get InfoSet Key using Utility Function ---<<<
        try:
            info_set_key = generate_info_set_key(game_state, acting_player_idx)
            if not info_set_key:
                raise ValueError("Key generation failed")
        except Exception as key_err:
            # if verbose: print(f"WARN _calculate_cfr: KeyGen Error P{acting_player_idx} depth {depth}: {key_err}")
            return 0.0 # Cannot proceed without a key

        # Get Available Actions (handling override/abstraction)
        available_actions = []
        try:
            if self.get_actions_override:
                available_actions = self.get_actions_override(game_state)
            else:
                raw_actions = game_state.get_available_actions()
                available_actions = ActionAbstraction.abstract_actions(raw_actions, game_state) if self.use_action_abstraction else raw_actions
            if not isinstance(available_actions, list):
                available_actions = []
        except Exception:
            return 0.0

        if not available_actions:
            utility = 0.0
            try:
                utility_val = game_state.get_utility(player_idx, initial_stacks)
                utility = float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
            except Exception:
                pass
            return utility

        # Get/Create InfoSet and Strategy
        try:
            info_set = self._get_or_create_info_set(info_set_key, available_actions)
            assert info_set # Make sure info_set is not None
            strategy = info_set.get_strategy()
        except Exception:
            return 0.0

        # Explore Actions Loop
        node_utility_perspective = 0.0
        action_utilities_perspective = {}
        for action in available_actions:
            action_prob = strategy.get(action, 0.0)
            if action_prob < 1e-9:
                action_utilities_perspective[action] = None
                continue
            try:
                next_game_state = game_state.apply_action(action)
            except Exception:
                action_utilities_perspective[action] = None
                continue

            next_reach_probs = reach_probs.copy()
            if acting_player_idx != player_idx: # Update opponent reach
                prob_factor=0.0
                current_reach=0.0
                if isinstance(action_prob,(int,float)) and not(np.isnan(action_prob)or np.isinf(action_prob)):
                    prob_factor = float(action_prob)
                if acting_player_idx < len(next_reach_probs) and isinstance(next_reach_probs[acting_player_idx],(int,float)) and not(np.isnan(next_reach_probs[acting_player_idx])or np.isinf(next_reach_probs[acting_player_idx])):
                    current_reach = float(next_reach_probs[acting_player_idx])
                updated_reach = np.clip(current_reach * prob_factor, 0.0, 1.0)
                next_reach_probs[acting_player_idx] = updated_reach

            try: # Recursive Call
                utility_from_action = self._calculate_cfr(
                    next_game_state,
                    next_reach_probs,
                    player_idx,
                    initial_stacks,
                    weight,
                    prune_threshold,
                    depth + 1,
                    verbose
                )
                action_utilities_perspective[action] = utility_from_action
                if isinstance(utility_from_action, (int, float)) and not (np.isnan(utility_from_action) or np.isinf(utility_from_action)):
                    node_utility_perspective += action_prob * utility_from_action
            except RecursionError as re_inner:
                raise re_inner
            except Exception:
                action_utilities_perspective[action] = None

        # Update Regrets/Strategy Sum if acting player is perspective player
        if acting_player_idx == player_idx:
            safe_reach = np.nan_to_num(reach_probs, nan=0.0, posinf=0.0, neginf=0.0)
            opp_reach_prod = 1.0
            if self.num_players > 1:
                opp_reaches = [safe_reach[p] for p in range(self.num_players) if p != player_idx]
                temp_prod = np.prod(opp_reaches) if opp_reaches else 1.0
                opp_reach_prod = float(temp_prod) if isinstance(temp_prod,(int,float)) and not (np.isnan(temp_prod) or np.isinf(temp_prod)) else 0.0

            player_reach_prob = 0.0
            if player_idx < len(safe_reach) and isinstance(safe_reach[player_idx],(int,float)) and not(np.isnan(safe_reach[player_idx])or np.isinf(safe_reach[player_idx])):
                 player_reach_prob = float(safe_reach[player_idx])

            node_util_val = 0.0
            if isinstance(node_utility_perspective, (int, float)) and not (np.isnan(node_utility_perspective) or np.isinf(node_utility_perspective)):
                 node_util_val = float(node_utility_perspective)

            # Skip updates if opponent reach is effectively zero
            if opp_reach_prod > 1e-12:
                for action in available_actions: # Regret Update
                    utility_a = action_utilities_perspective.get(action)
                    if utility_a is None or not isinstance(utility_a, (int, float)) or np.isnan(utility_a) or np.isinf(utility_a):
                        continue

                    instant_regret = utility_a - node_util_val
                    if np.isnan(instant_regret) or np.isinf(instant_regret):
                        continue

                    current_regret_sum = info_set.regret_sum.get(action, 0.0)
                    current_regret_sum = float(current_regret_sum) if isinstance(current_regret_sum,(int,float)) and not(np.isnan(current_regret_sum)or np.isinf(current_regret_sum)) else 0.0

                    regret_increment = opp_reach_prod * instant_regret
                    updated_regret_sum = current_regret_sum
                    if not (np.isnan(regret_increment) or np.isinf(regret_increment)):
                        updated_regret_sum += regret_increment

                    new_regret_value = max(0.0, updated_regret_sum)
                    info_set.regret_sum[action] = new_regret_value

                # Strategy Sum Update (Linear CFR: pi_{i} * T)
                strategy_sum_weight = player_reach_prob * weight
                if not (np.isnan(strategy_sum_weight) or np.isinf(strategy_sum_weight)):
                    info_set.update_strategy_sum(strategy, strategy_sum_weight)

        # Return Node EV for Perspective Player
        final_utility = float(node_utility_perspective) if isinstance(node_utility_perspective, (int, float)) and not (np.isnan(node_utility_perspective) or np.isinf(node_utility_perspective)) else 0.0
        return final_utility

    # --- REMOVE _create_info_set_key (using utility now) ---
    # def _create_info_set_key(self, game_state, player_idx):
    #     pass # Removed

    # --- _get_or_create_info_set (unchanged, kept) ---
    def _get_or_create_info_set(self, key, actions):
        if not isinstance(key, str) or not key:
            return None
        if key not in self.information_sets:
            valid_actions = []
            seen_action_repr = set()
            if not isinstance(actions, list):
                actions = []
            for action in actions:
                action_tuple = None
                try:
                    if isinstance(action, tuple) and len(action) == 2:
                        action_tuple = (str(action[0]), int(round(float(action[1]))))
                    elif isinstance(action, str) and action in ['fold', 'check']:
                        action_tuple = (action, 0)
                except (ValueError, TypeError):
                    continue

                if action_tuple and action_tuple not in seen_action_repr:
                    valid_actions.append(action_tuple)
                    seen_action_repr.add(action_tuple)

            if valid_actions:
                try:
                    self.information_sets[key] = InformationSet(valid_actions)
                except Exception as e:
                    print(f"ERROR creating InfoSet '{key}': {e}")
                    return None
            else:
                return None
        return self.information_sets.get(key)

    # --- get_strategy (unchanged, kept) ---
    def get_strategy(self):
        average_strategy_map = {}
        num_total_sets = len(self.information_sets)
        if num_total_sets == 0:
            return {}

        num_invalid_sets = 0
        items_iterable = tqdm(self.information_sets.items(), desc="AvgStrat", total=num_total_sets, disable=(num_total_sets < 10000), unit="set") if num_total_sets > 10000 else self.information_sets.items()

        for key, info_set_obj in items_iterable:
            if not isinstance(info_set_obj, InformationSet):
                num_invalid_sets += 1
                continue
            try:
                avg_strat_for_set = info_set_obj.get_average_strategy()
                if isinstance(avg_strat_for_set, dict):
                    prob_sum = sum(avg_strat_for_set.values())
                    if abs(prob_sum - 1.0) < 0.01 or abs(prob_sum) < 1e-6:
                        if all(isinstance(k, tuple) and len(k) == 2 for k in avg_strat_for_set.keys()):
                            average_strategy_map[key] = avg_strat_for_set
                        else:
                            num_invalid_sets += 1
                    else:
                        num_invalid_sets += 1
                elif isinstance(avg_strat_for_set, dict) and not avg_strat_for_set:
                    # Handle empty strategy dictionaries if needed (e.g., terminal nodes before action)
                    average_strategy_map[key] = {}
                else:
                    num_invalid_sets += 1
            except Exception:
                num_invalid_sets += 1
        # Optionally print warning about invalid sets
        # if num_invalid_sets > 0: print(f"WARN get_strategy: Skipped {num_invalid_sets}/{num_total_sets} invalid/malformed info sets.")
        return average_strategy_map

    # --- save/load methods (unchanged, kept) ---
    def _save_final_strategy(self, output_directory, strategy_map):
        if not output_directory:
            return
        final_path = os.path.join(output_directory, "final_strategy.pkl")
        try:
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
        except OSError:
            pass # Directory likely already exists
        try:
            with open(final_path, 'wb') as f:
                pickle.dump(strategy_map, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\nFinal average strategy saved: {final_path} ({len(strategy_map):,} sets)")
        except Exception as e:
            print(f"\nERROR saving final strategy to {final_path}: {e}")

    def _save_checkpoint(self, output_directory, current_iteration):
        if not output_directory:
            return
        checkpoint_data = {
            'iterations': current_iteration,
            'information_sets': self.information_sets,
            'num_players': self.num_players,
            'use_card_abstraction': self.use_card_abstraction,
            'use_action_abstraction': self.use_action_abstraction,
            'training_start_time': self.training_start_time
        }
        checkpoint_path = os.path.join(output_directory, f"cfr_checkpoint_{current_iteration}.pkl")
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        except OSError:
             pass # Directory likely already exists
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Don't print here, handled by the main loop's log message
        except Exception as e:
            print(f"\nERROR saving checkpoint to {checkpoint_path}: {e}")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint file not found: {checkpoint_path}.")
            return False
        try:
            print(f"Loading checkpoint from: {checkpoint_path}...")
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)

            self.iterations = data.get('iterations', 0)
            loaded_sets = data.get('information_sets', {})
            if isinstance(loaded_sets, dict):
                self.information_sets = loaded_sets
            else:
                print("ERROR: Checkpoint 'information_sets' not dict.")
                return False

            self.num_players = data.get('num_players', self.num_players)
            # Reload abstraction flags from checkpoint
            self.use_card_abstraction = data.get('use_card_abstraction', self.use_card_abstraction)
            self.use_action_abstraction = data.get('use_action_abstraction', self.use_action_abstraction)
            self.training_start_time = data.get('training_start_time', time.time()) # Default to now if missing
            self.get_actions_override = None # Cannot restore function reference

            print(f"Checkpoint loaded. Resuming from iter {self.iterations + 1}. ({len(self.information_sets):,} sets)")
            return True
        except Exception as e:
            print(f"ERROR loading checkpoint: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False

# --- END OF FILE organized_poker_bot/cfr/cfr_trainer.py ---
