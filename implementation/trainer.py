import os
import pickle
import numpy as np
import time

from infoset import InfoSet
from implementation.buckets.action import ActionBucket
from info_set_util import generate_key

class Trainer:

    def __init__(self, game_state_class, num_players=2, ):
        self.game_state_class = game_state_class
        self.num_players = num_players
        self.information_sets = {}
        self.iterations = 0
        self.training_start_time = None

    def train(self, iterations=1000, checkpoint_freq=100, output_dir=None):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.training_start_time = self.training_start_time or time.time()
        
        start_iter = self.iterations
        end_iter = start_iter + iterations

        for i in range(start_iter, end_iter):
            current_iter_num = i + 1
            
            game_state = self.game_state_class(self.num_players)
            initial_stacks = [10000.0] * self.num_players
            
            dealer_pos = current_iter_num % self.num_players
            game_state.start_new_hand(dealer_pos=dealer_pos, player_stacks=initial_stacks)
            if game_state.is_terminal() or game_state.current_player_idx == -1:
                continue

            reach_probs = np.ones(self.num_players)
            for p_idx in range(self.num_players):
                self.calculate_cfr(
                    game_state.clone(),
                    reach_probs.copy(),
                    p_idx,
                    initial_stacks,
                    float(current_iter_num),
                    0.0, 0
                )

            self.iterations = current_iter_num
            if output_dir and (current_iter_num % checkpoint_freq == 0):
                self.save_checkpoint(output_dir, current_iter_num)
        
        final_strat = self.get_strategy()
        if output_dir:
            self.save_final_strategy(output_dir, final_strat)
        
        return final_strat
    
    def calculate_cfr(self, game_state, reach_probs, player_idx, initial_stacks, weight, depth):
        if game_state.is_terminal():
            return game_state.get_utility(player_idx, initial_stacks)
            
        if depth > 3000:
            return 0.0

        curr_player = game_state.current_player_idx
        
        is_folded = game_state.player_folded[curr_player]
        is_all_in = game_state.player_all_in[curr_player]

        if is_folded or is_all_in:
            temp_state = game_state.clone()
            original_turn_idx = temp_state.current_player_idx
            temp_state.rotate_turn()
            
            if temp_state.current_player_idx == original_turn_idx or temp_state.is_terminal():
                return temp_state.get_utility(player_idx, initial_stacks)
            else:
                return self.calculate_cfr(temp_state, reach_probs, player_idx, initial_stacks, weight, depth + 1)

        # Get InfoSet Key 
        info_set_key = generate_key(game_state, curr_player)
        
        # Get Available Actions
        raw_actions = game_state.get_available_actions()
        available_actions = ActionBucket.abstract_actions(raw_actions, game_state)
        
        # Create InfoSet and get strategy
        info_set = self.get_info_set(info_set_key, available_actions)
        strategy = info_set.get_strategy()

        # Explore Actions and calculate node utility
        node_utility_perspective = 0.0
        action_utilities_perspective = {}
        
        for action in available_actions:
            action_prob = strategy.get(action, 0.0)
            if action_prob < 1e-9:  # Keep this numerical check
                continue
                
            next_game_state = game_state.apply_action(action)

            next_reach_probs = reach_probs.copy()
            if curr_player != player_idx:  # Update opponent reach
                next_reach_probs[curr_player] *= action_prob

            # Recursive Call
            utility_from_action = self.calculate_cfr(
                next_game_state,
                next_reach_probs,
                player_idx,
                initial_stacks,
                weight,
                depth + 1
            )
            
            action_utilities_perspective[action] = utility_from_action
            node_utility_perspective += action_prob * utility_from_action

        # Update Regrets/Strategy Sum if acting player is perspective player
        if curr_player == player_idx:
            # Calculate opponent reach product
            opp_reach_prod = 1.0
            if self.num_players > 1:
                opp_reaches = [reach_probs[p] for p in range(self.num_players) if p != player_idx]
                opp_reach_prod = np.prod(opp_reaches)

            player_reach_prob = reach_probs[player_idx]

            # Keep numerical stability check
            if opp_reach_prod > 1e-12:
                for action in available_actions:
                    utility_a = action_utilities_perspective.get(action)
                    instant_regret = utility_a - node_utility_perspective
                    
                    # Update regret sum
                    info_set.regret_sum[action] = max(0.0, info_set.regret_sum.get(action, 0.0) + opp_reach_prod * instant_regret)

                # Update strategy sum (Linear CFR: pi_{i} * T)
                info_set.update_strategy_sum(strategy, player_reach_prob * weight)

        return node_utility_perspective
    
    def get_info_set(self, key, actions):
        # Return existing info set if available
        if key in self.information_sets:
            return self.information_sets[key]
            
        # Create a new info set
        valid_actions = []
        seen_actions = set()
        
        for action in actions:
            action_tuple = None
            
            # Format action tuple consistently
            if isinstance(action, tuple) and len(action) == 2:
                action_tuple = (str(action[0]), int(round(float(action[1]))))
            elif isinstance(action, str) and action in ['fold', 'check']:
                action_tuple = (action, 0)
            else:
                continue
                
            # Add unique actions
            if action_tuple not in seen_actions:
                valid_actions.append(action_tuple)
                seen_actions.add(action_tuple)

        self.information_sets[key] = InfoSet(valid_actions)
        return self.information_sets[key]
    
    def get_strategy(self):
        average_strategy_map = {}
        
        # Process each info set
        for key, info_set_obj in self.information_sets.items():
            # Get average strategy and add to map
            avg_strat = info_set_obj.get_average_strategy()
            average_strategy_map[key] = avg_strat
        
        return average_strategy_map
    
    def save_final_strategy(self, output_directory, strategy_map):
        if not output_directory:
            return

        final_path = os.path.join(output_directory, "final_strategy.pkl")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        
        with open(final_path, 'wb') as f:
            pickle.dump(strategy_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_checkpoint(self, output_directory, current_iteration):
        if not output_directory:
            return
            
        checkpoint_data = {
            'iterations': current_iteration,
            'information_sets': self.information_sets,
            'num_players': self.num_players,
            'training_start_time': self.training_start_time
        }
        
        checkpoint_path = os.path.join(output_directory, f"cfr_checkpoint_{current_iteration}.pkl")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        self.iterations = data.get('iterations', 0)
        self.information_sets = data.get('information_sets', {})
        self.num_players = data.get('num_players', self.num_players)
        self.training_start_time = data.get('training_start_time', time.time())
        
        return True