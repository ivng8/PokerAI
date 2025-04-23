import numpy as np

from implementation.players.player import Player
from implementation.algo.strategy import MCCFRStrategy
from implementation.algo.dls import DepthLimitedSearch

class BotPlayer(Player):

    def __init__(self, strategy, name="Bot", stack=10000, use_depth_limited_search=False, search_depth=1, search_iterations=100):
        super().__init__(name=name, stack=stack, is_human=False, is_random=False)

        # Initialize BotPlayer specific attributes
        self.strategy_obj = None
        if isinstance(strategy, MCCFRStrategy):
            self.strategy_obj = strategy
        elif isinstance(strategy, dict):
            self.strategy_obj = MCCFRStrategy()
            self.strategy_obj.strategy = strategy

        self.use_depth_limited_search = use_depth_limited_search
        self.search_depth = search_depth
        self.search_iterations = search_iterations
        self.dls = None

        # Initialize depth-limited search if enabled and strategy is valid
        if self.use_depth_limited_search and self.strategy_obj and self.strategy_obj.strategy:
            self.dls = DepthLimitedSearch(
                self.strategy_obj,
                search_depth=self.search_depth,
                num_iterations=self.search_iterations
            )
        else:
            self.use_depth_limited_search = False

    def get_action(self, game_state, player_idx):
        # 1. Use DLS if enabled and available
        if self.use_depth_limited_search and self.dls and not game_state.is_terminal():
            # Estimate initial stacks for DLS's utility calculation
            initial_stacks_estimation = [0.0] * game_state.num_players
            for i in range(game_state.num_players):
                current_stack = float(game_state.player_stacks[i]) if i < len(game_state.player_stacks) and not np.isnan(game_state.player_stacks[i]) else 0.0
                total_invested = float(game_state.player_total_bets_in_hand[i]) if i < len(game_state.player_total_bets_in_hand) and not np.isnan(game_state.player_total_bets_in_hand[i]) else 0.0
                initial_stacks_estimation[i] = current_stack + total_invested

            # Get action using DLS
            action = self.dls.get_action(game_state, player_idx, initial_stacks_estimation)
            if isinstance(action, tuple) and len(action) == 2:
                return action

        # 2. Use Blueprint Strategy if DLS not used or failed
        if self.strategy_obj and self.strategy_obj.strategy:
            action = self.strategy_obj.get_action(game_state, player_idx)
            if isinstance(action, tuple) and len(action) == 2:
                return action

        # 3. Default Action (if all else fails)
        available_actions = game_state.get_available_actions()
        if ('check', 0) in available_actions:
            return ('check', 0)
        if ('fold', 0) in available_actions:
            return ('fold', 0)
        
        return available_actions[0] if available_actions else ('fold', 0)
    
    def update_strategy(self, new_strategy):
        if isinstance(new_strategy, MCCFRStrategy):
            self.strategy_obj = new_strategy
        elif isinstance(new_strategy, dict):
            self.strategy_obj = MCCFRStrategy()
            self.strategy_obj.strategy = new_strategy
        else:
            return

        # Update depth-limited search if enabled
        if self.use_depth_limited_search and self.strategy_obj and self.strategy_obj.strategy:
            self.dls = DepthLimitedSearch(
                self.strategy_obj,
                search_depth=self.search_depth,
                num_iterations=self.search_iterations
            )
        else:
            self.dls = None
            self.use_depth_limited_search = False