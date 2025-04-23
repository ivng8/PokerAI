import random
import numpy as np

from info_set_util import generate_key

class DepthLimitedSearch:

    def __init__(self, blueprint_strategy, search_depth=1, num_iterations=100, exploration_constant=1.414):
        self.blueprint_strategy = blueprint_strategy
        self.search_depth = max(1, search_depth)
        self.num_iterations = max(10, num_iterations)
        self.exploration_constant = exploration_constant

        self.node_visits = {}
        self.action_values = {}
        self.action_visits = {}

    def get_action(self, game_state, player_idx, initial_stacks):
        if game_state.is_terminal():
            available_actions = game_state.get_available_actions()
            if available_actions:
                return available_actions[0]
            else:
                ('fold', 0)

        self.node_visits = {}
        self.action_values = {}
        self.action_visits = {}

        root_info_set_key = generate_key(game_state, player_idx)
        self.node_visits[root_info_set_key] = 0
        available_actions = game_state.get_available_actions()
        
        if not available_actions:
            return ('fold', 0)
            
        if len(available_actions) == 1:
            return available_actions[0]

        for _ in range(self.num_iterations):
            sim_state = game_state.clone()
            self.simulate(sim_state, player_idx, self.search_depth, initial_stacks)

        # Choose Best Action (Most Visited)
        best_action = None
        max_visits = -1

        for action in available_actions:
            action_key = self.get_action_key(root_info_set_key, action)
            if action_key is None:
                continue
                
            visits = self.action_visits.get(action_key, 0)
            if visits > max_visits:
                max_visits = visits
                best_action = action

        # Fallback if no visits
        if best_action is None:
            best_action = self.blueprint_strategy.get_action(game_state, player_idx)
            
            # Ensure tuple format
            if isinstance(best_action, str):
                best_action = (best_action, 0)
                
            # If not in available actions, use default strategy
            if best_action not in available_actions:
                best_action = self.blueprint_strategy.default_strategy(available_actions)

        # Ensure chosen action is available
        if best_action not in available_actions:
            if ('check', 0) in available_actions:
                return ('check', 0)
                
            call_actions = [a for a in available_actions if a[0] == 'call']
            if call_actions:
                return call_actions[0]
                
            if ('fold', 0) in available_actions:
                return ('fold', 0)
                
            return available_actions[0]

        return best_action
    
    def simulate(self, sim_state, player_idx_perspective, depth, initial_stacks):
        # Base Cases: Terminal State
        if sim_state.is_terminal():
            utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
            return float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0

        # Max Depth Reached
        if depth <= 0:
            utility = self.blueprint_rollout(sim_state, player_idx_perspective, initial_stacks)
            return float(utility) if isinstance(utility, (int, float)) and not (np.isnan(utility) or np.isinf(utility)) else 0.0

        # Identify acting player
        current_player_idx = sim_state.current_player_idx
        if not (0 <= current_player_idx < sim_state.num_players):
            return 0.0

        # Handle inactive players by advancing state
        while True:
            is_player_valid = (0 <= current_player_idx < sim_state.num_players)
            is_player_active = False
            
            if is_player_valid:
                is_folded = sim_state.player_folded[current_player_idx] if current_player_idx < len(sim_state.player_folded) else True
                is_all_in = sim_state.player_all_in[current_player_idx] if current_player_idx < len(sim_state.player_all_in) else True
                is_player_active = not is_folded and not is_all_in

            if not is_player_valid or not is_player_active:
                if sim_state.is_terminal():
                    utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
                    return float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0

                original_idx = current_player_idx
                sim_state.rotate_turn()
                current_player_idx = sim_state.current_player_idx

                if current_player_idx == original_idx or sim_state.is_terminal():
                    utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
                    return float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
            else:
                break

        # Generate info set key
        info_set_key = generate_key(sim_state, current_player_idx)
        if not info_set_key:
            return 0.0

        # Selection / Expansion
        node_visit_count = self.node_visits.get(info_set_key, -1)

        if node_visit_count == -1:  # Expand new node
            self.node_visits[info_set_key] = 0
            value = self.blueprint_rollout(sim_state.clone(), player_idx_perspective, initial_stacks)
            value = float(value) if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)) else 0.0
            self.node_visits[info_set_key] = 1
            return value
        else:  # Node visited before, use UCB
            available_actions = sim_state.get_available_actions()
            
            if not available_actions:
                utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
                return float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0

            # Select action using UCB1
            chosen_action = self.select_action_ucb(info_set_key, available_actions)
            if not isinstance(chosen_action, tuple) or chosen_action not in available_actions:
                chosen_action = self.blueprint_strategy.default_strategy(available_actions)

        # Apply chosen action
        next_sim_state = sim_state.apply_action(chosen_action)

        # Recursive simulation
        value = self.simulate(next_sim_state, player_idx_perspective, depth - 1, initial_stacks)
        value = float(value) if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)) else 0.0

        # Backpropagation
        action_key = self.get_action_key(info_set_key, chosen_action)
        if action_key:
            if action_key not in self.action_visits:
                self.action_visits[action_key] = 0
                self.action_values[action_key] = 0.0
                
            self.action_visits[action_key] += 1
            self.action_values[action_key] += value
            self.node_visits[info_set_key] = self.node_visits.get(info_set_key, 0) + 1

        return value
    
    def select_action_ucb(self, info_set_key, available_actions):
        parent_visits = self.node_visits.get(info_set_key, 1)
        log_parent_visits = np.log(max(1, parent_visits))

        if not available_actions:
            return ('fold', 0)

        unvisited_actions = []
        for action in available_actions:
            action_key = self.get_action_key(info_set_key, action)
            if not action_key:
                continue
                
            if self.action_visits.get(action_key, 0) == 0:
                unvisited_actions.append(action)

        if unvisited_actions:
            return random.choice(unvisited_actions)

        best_action = None
        best_ucb_score = float('-inf')

        for action in available_actions:
            action_key = self.get_action_key(info_set_key, action)
            if not action_key:
                continue
                
            action_visit_count = self.action_visits.get(action_key, 1)
            action_total_value = self.action_values.get(action_key, 0.0)

            average_value = action_total_value / action_visit_count
            exploration_term = self.exploration_constant * np.sqrt(log_parent_visits / action_visit_count)
            ucb_score = average_value + exploration_term

            if ucb_score > best_ucb_score:
                best_ucb_score = ucb_score
                best_action = action

        return best_action if best_action else random.choice(available_actions)
    
    def blueprint_rollout(self, sim_state, player_idx_perspective, initial_stacks):
        rollout_depth = 0
        max_rollout_depth = 30

        while not sim_state.is_terminal() and rollout_depth < max_rollout_depth:
            current_player_idx = sim_state.current_player_idx

            # Check if player is active
            is_player_valid = (0 <= current_player_idx < sim_state.num_players)
            is_player_active = False
            
            if is_player_valid:
                is_folded = sim_state.player_folded[current_player_idx] if current_player_idx < len(sim_state.player_folded) else True
                is_all_in = sim_state.player_all_in[current_player_idx] if current_player_idx < len(sim_state.player_all_in) else True
                is_player_active = not is_folded and not is_all_in

            if not is_player_valid or not is_player_active:
                original_idx = current_player_idx
                sim_state.rotate_turn()
                
                if sim_state.current_player_idx == original_idx or sim_state.is_terminal():
                    break
                continue

            # Get action for active player
            available_rollout = sim_state.get_available_actions()
            if not available_rollout:
                break

            action = self.blueprint_strategy.get_action(sim_state, current_player_idx)

            # Format and validate action
            if isinstance(action, str) and action in ['fold', 'check']:
                action = (action, 0)
                
            if not isinstance(action, tuple) or action not in available_rollout:
                action = self.blueprint_strategy.default_strategy(available_rollout)

            # Apply action
            sim_state = sim_state.apply_action(action)
            rollout_depth += 1

        # Return final utility
        if sim_state.is_terminal():
            utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
            return float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
        else:
            utility_val = sim_state.get_utility(player_idx_perspective, initial_stacks)
            return float(utility_val) if isinstance(utility_val, (int, float)) and not (np.isnan(utility_val) or np.isinf(utility_val)) else 0.0
        
    def get_action_key(self, info_set_key, action):
        if not info_set_key or not isinstance(info_set_key, str):
            return None

        action_tuple = None
        if isinstance(action, str) and action in ['fold', 'check']:
            action_tuple = (action, 0)
        elif isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], str):
            # Convert action amount directly, assuming it will be valid
            amount = int(round(float(action[1])))
            action_tuple = (action[0], amount)
        else:
            return None

        action_str = f"{action_tuple[0]}_{action_tuple[1]}"
        return f"{info_set_key}|A:{action_str}"