import pickle
import numpy as np
import os

from game_state import GameState
from info_set_util import generate_key

class MCCFRStrategy:
    
    def __init__(self):
        self.strategy = {}

    def get_action(self, game_state, player_idx):
        info_set_key = generate_key(game_state, player_idx)
    
        # Strategy Lookup - Use .get() for safer dictionary access
        action_probs = self.strategy.get(info_set_key)

        # If info set not found in strategy map, or invalid, use default
        if action_probs is None or not isinstance(action_probs, dict) or not action_probs:
            # InfoSet not found or invalid format - Use default strategy
            available_actions = game_state.get_available_actions()
            return self.default_strategy(game_state, available_actions)

        # Choose Action based on Loaded Probabilities
        return self.choose_action(action_probs, game_state)
    
    def choose_action(self, action_probs, game_state):
        # Get currently available actions from the game state
        available_actions = game_state.get_available_actions()
        
        if not available_actions:
            return ('fold', 0)  # Default safe action

        # Filter the strategy probabilities to include only available actions
        valid_actions_dict = {}
        total_prob_available = 0.0
        
        for action_tuple, prob in action_probs.items():
            if action_tuple in available_actions and isinstance(prob, (int, float)) and prob > 1e-9:
                valid_actions_dict[action_tuple] = prob
                total_prob_available += prob

        # If no available actions match the strategy (or sum is zero), use default
        if not valid_actions_dict or total_prob_available <= 1e-9:
            return self.default_strategy(game_state, available_actions)

        # Normalize probabilities of available actions
        normalized_probs = []
        action_list = list(valid_actions_dict.keys())
        
        for action in action_list:
            prob = valid_actions_dict.get(action, 0.0)
            normalized_probs.append(prob / total_prob_available)

        # Choose action stochastically based on normalized probabilities
        chosen_action_index = np.random.choice(len(action_list), p=normalized_probs)
        chosen_action = action_list[chosen_action_index]
        
        return chosen_action
    
    def _default_strategy(self, game_state, available_actions):
        if not available_actions or not isinstance(available_actions, list):
            return ('fold', 0)

        # Ensure actions are in the expected tuple format if possible
        valid_formatted_actions = []
        for act in available_actions:
            if isinstance(act, tuple) and len(act) == 2:
                valid_formatted_actions.append(act)
        available_actions = valid_formatted_actions

        if not available_actions:
            return ('fold', 0)

        # Check if check is available
        check_action = ('check', 0)
        if check_action in available_actions:
            return check_action

        # Check if call is available (find any call action, regardless of amount)
        call_actions = [a for a in available_actions if isinstance(a[0], str) and a[0].lower() == 'call']
        if call_actions:
            return call_actions[0]

        # If check/call not available, default to folding (safest)
        fold_action = ('fold', 0)
        if fold_action in available_actions:
            return fold_action

        # If somehow fold is also unavailable, return the first action in the list
        return available_actions[0]
    
    def save(self, filename):
        # Ensure directory exists
        dir_name = os.path.dirname(filename)
        if dir_name:  # Only create if path includes a directory
            os.makedirs(dir_name, exist_ok=True)
        
        # Save using highest protocol
        with open(filename, 'wb') as f:
            pickle.dump(self.strategy, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Strategy saved successfully to {filename}")