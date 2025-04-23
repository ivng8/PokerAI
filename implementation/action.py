from game_state import GameState

class ActionBucket:
    POT_FRACTIONS_BET = [0.33, 0.5, 0.75, 1.0, 1.25]
    RAISE_MULTIPLIERS = [2.5, 3.5]

    @staticmethod
    def abstract_actions(available_actions, game_state):
        abstracted_actions_dict = {}

        player_idx = game_state.current_player_idx
        if player_idx < 0 or player_idx >= game_state.num_players: 
            return []

        player_stack = game_state.player_stacks[player_idx]
        player_bet_this_round = game_state.player_bets_in_round[player_idx]
        current_bet_level = game_state.current_bet
        pot_size = game_state.pot
        last_raise_amount = max(game_state.last_raise, game_state.big_blind)

        # 1. Add essential non-aggressive actions
        if ('fold', 0) in available_actions: 
            abstracted_actions_dict[('fold', 0)] = ('fold', 0)
        
        if ('check', 0) in available_actions: 
            abstracted_actions_dict[('check', 0)] = ('check', 0)
        
        call_action = next((a for a in available_actions if a[0] == 'call'), None)
        if call_action: 
            abstracted_actions_dict[call_action] = call_action

        # 2. Determine action type and aggression possibility
        can_bet = any(a[0] == 'bet' for a in available_actions)
        can_raise = any(a[0] == 'raise' for a in available_actions)
        allow_aggression = (game_state.raise_count_this_street < game_state.MAX_RAISES_PER_STREET) and (can_bet or can_raise)
        action_type = "bet" if current_bet_level < 0.01 else "raise"

        if not allow_aggression:
            return sorted(list(abstracted_actions_dict.values()), 
                        key=lambda a: ({"fold":0,"check":1,"call":2}[a[0]], a[1]))

        # --- Aggression Possible ---
        # Find min/max aggressive actions TO amounts
        original_min_aggressive_to = float('inf')
        original_max_aggressive_to = 0.0
        
        for act, amt in available_actions:
            if act in ['bet', 'raise']:
                original_min_aggressive_to = min(original_min_aggressive_to, amt)
                original_max_aggressive_to = max(original_max_aggressive_to, amt)

        # 3. Add MIN-LEGAL aggressive action
        min_legal_amount = int(round(original_min_aggressive_to))
        cost_min_legal = min_legal_amount - player_bet_this_round
        if cost_min_legal <= player_stack + 0.01:
            min_legal_action_tuple = (action_type, min_legal_amount)
            abstracted_actions_dict[min_legal_action_tuple[1]] = min_legal_action_tuple

        # 4. Add specific sizings based on action type
        if action_type == 'bet':
            # --- BET SIZING: Use Pot Fractions ---
            for fraction in ActionBucket.POT_FRACTIONS_BET:
                target_total_bet = pot_size * fraction
                
                # Clamp by min legal bet and max possible bet
                target_total_bet = max(original_min_aggressive_to, target_total_bet)
                target_total_bet = min(original_max_aggressive_to, target_total_bet)
                
                cost_to_reach_target = target_total_bet - player_bet_this_round
                if cost_to_reach_target <= player_stack + 0.01 and cost_to_reach_target > 0.01:
                    action_tuple = (action_type, int(round(target_total_bet)))
                    abstracted_actions_dict[action_tuple[1]] = action_tuple
        else:
            # --- RAISE SIZING: Use Multipliers of Last Raise ---
            for multiplier in ActionBucket.RAISE_MULTIPLIERS:
                raise_increment_amount = multiplier * last_raise_amount
                target_total_bet = current_bet_level + raise_increment_amount
                
                # Clamp by min legal raise and max possible raise
                target_total_bet = max(original_min_aggressive_to, target_total_bet)
                target_total_bet = min(original_max_aggressive_to, target_total_bet)
                
                cost_to_reach_target = target_total_bet - player_bet_this_round
                if cost_to_reach_target <= player_stack + 0.01 and cost_to_reach_target > 0.01:
                    action_tuple = (action_type, int(round(target_total_bet)))
                    abstracted_actions_dict[action_tuple[1]] = action_tuple

        # 5. Add ALL-IN aggressive action if valid & distinct
        if original_max_aggressive_to > 0.01 and not (action_type == 'raise' and original_max_aggressive_to <= current_bet_level + 0.01):
            all_in_action_tuple = (action_type, int(round(original_max_aggressive_to)))
            abstracted_actions_dict[all_in_action_tuple[1]] = all_in_action_tuple

        # Sort final list
        def sort_key(a): 
            t,amt = a
            o = {"fold":0,"check":1,"call":2,"bet":3,"raise":4}
            return (o.get(t,99), amt)
            
        return sorted(list(abstracted_actions_dict.values()), key=sort_key)