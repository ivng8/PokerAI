import random

from implementation.game.game_state import GameState

POS_NAMES = ["BTN", "SB", "BB", "LJ", "HJ", "CO"]

class PokerGame:

    def __init__(self, players, small_blind=50, big_blind=100, interactive=False):
        for i, p in enumerate(players):
            if not hasattr(p, 'stack'): p.stack = 10000
            if not hasattr(p, 'name'): p.name = f"Player_{i}"
            p.position = i

        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_position = random.randint(0, len(players) - 1)
        self.game_state = None
        self.hand_history = []
        self.interactive = interactive

    def get_pos_name(self, player_idx, dealer_idx, num_players):
        if num_players == 6:
            pos = (player_idx - dealer_idx) % num_players
            return POS_NAMES[pos]
        if num_players == 2:
            return POS_NAMES[1 - ((player_idx - dealer_idx) % 2)]
        return f"Pos{player_idx}"
        
    def run(self, num_hands=10):
        for _ in range(num_hands):
            self.rotate_dealer()
            self.play_hand()
            
            players_with_chips = [p for p in self.players if hasattr(p, 'stack') and p.stack > 0]
            if len(players_with_chips) <= 1:
                break
            
    def play_hand(self):
        current_stacks = [getattr(p, 'stack', 0) for p in self.players]
        active_player_indices = [i for i, stack in enumerate(current_stacks) if stack > 0]

        if len(active_player_indices) < 2:
            for i, player in enumerate(self.players): 
                player.stack = current_stacks[i]
            return

        self.game_state = GameState(len(self.players), 0, self.small_blind, self.big_blind)
        self.game_state.player_stacks = current_stacks
        self.game_state.dealer_position = self.dealer_position
        self.game_state.active_players = active_player_indices

        self.game_state.deal_hole()

        # Keep minimal interactive mode functionality but remove prints
        if self.interactive:
            for i, player in enumerate(self.players):
                if hasattr(player, 'human') and player.human and i in self.game_state.active_players:
                    pass

        self.game_state.put_blinds()

        if len([p for p in self.game_state.active_players if self.game_state.player_stacks[p] > 0]) > 1:
            self.betting_round("Preflop")

        if len(self.game_state.active_players) > 1:
            self.game_state.deal_flop()
            self.betting_round("Flop")

        if len(self.game_state.active_players) > 1:
            self.game_state.deal_turn()
            self.betting_round("Turn")

        if len(self.game_state.active_players) > 1:
            self.game_state.deal_river()
            self.betting_round("River")

        self.conclude_hand()

        for i, player in enumerate(self.players):
            if i < len(self.game_state.player_stacks): 
                player.stack = self.game_state.player_stacks[i]

    def betting_round(self, round_name):
        active_players_can_act = [p_idx for p_idx in self.game_state.active_players if self.game_state.player_stacks[p_idx] > 0]
        if len(active_players_can_act) <= 1:
            return

        # Set starting player if needed
        if self.game_state.current_player_idx not in self.game_state.active_players or self.game_state.player_stacks[self.game_state.current_player_idx] <= 0:
            start_search = (self.dealer_position + 1) % len(self.players) if round_name != "Preflop" else (self.dealer_position + 3) % len(self.players)
            initial_player = start_search
            for _ in range(len(self.players)):
                if initial_player < len(self.players) and initial_player in self.game_state.active_players and self.game_state.player_stacks[initial_player] > 0:
                    self.game_state.current_player_idx = initial_player
                    break
                initial_player = (initial_player + 1) % len(self.players)

        num_actions_this_round = 0
        max_actions = len(self.players) * 4

        while not self.game_state.betting_done() and num_actions_this_round < max_actions:
            player_idx = self.game_state.current_player_idx

            if player_idx not in self.game_state.active_players or self.game_state.player_stacks[player_idx] <= 0:
                self.game_state.rotate_turn()
                if self.game_state.current_player_idx == player_idx:
                    break  # Avoid infinite loop
                continue

            player = self.players[player_idx]
            available_actions = self.game_state.get_available_actions()

            if not available_actions:
                self.game_state.rotate_turn()
                if self.game_state.current_player_idx == player_idx:
                    break
                continue

            action = self.get_player_action(player, player_idx, available_actions, self.game_state)

            # Default action if invalid format
            if not isinstance(action, tuple) or len(action) != 2:
                check_action = ('check', 0)
                fold_action = ('fold', 0)
                if check_action in available_actions:
                    action = check_action
                elif fold_action in available_actions:
                    action = fold_action
                else:
                    action = available_actions[0] if available_actions else ('fold', 0)

            self.game_state = self.game_state.apply_action(action)
            num_actions_this_round += 1

    def get_player_action(self, player, player_idx, available_actions, current_game_state):        
        if hasattr(player, 'get_action') and callable(player.get_action):
            action = player.get_action(current_game_state, player_idx)
        elif hasattr(player, 'human') and player.human and self.interactive:
            action = self.get_human_action(player, available_actions, current_game_state)
        elif hasattr(player, 'random') and player.random and available_actions:
            action = random.choice(available_actions)
        
        # Default action if needed
        if not (isinstance(action, tuple) and len(action) == 2):
            check_action = ('check', 0)
            fold_action = ('fold', 0)
            
            if check_action in available_actions:
                action = check_action
            elif fold_action in available_actions:
                action = fold_action
            elif available_actions:
                action = available_actions[0]
            else:
                action = ('fold', 0)
                
        return action


    def get_human_action(self, available_actions):
        action_map = {}
        
        for i, action_tuple in enumerate(available_actions):
            action_map[i+1] = action_tuple
        
        while True:
            try:
                choice_str = input("Enter choice number: ")
                choice_num = int(choice_str)
                if choice_num in action_map:
                    return action_map[choice_num]
            except ValueError:
                pass
            except EOFError:
                return ('fold', 0)

    def conclude_hand(self):
        final_active_players = self.game_state.active_players.copy()

        if len(final_active_players) == 1:
            winner_idx = final_active_players[0]
            amount_won = self.game_state.pot
            self.game_state.player_stacks[winner_idx] += amount_won
            self.game_state.pot = 0
        elif len(final_active_players) > 1:
            # Pass the main player list for names
            self.game_state.determine_winners(self.players)
        else:
            self.game_state.pot = 0

    def rotate_dealer(self):
        self.dealer_position = (self.dealer_position + 1) % len(self.players)
