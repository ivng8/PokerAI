import random

from game_state import GameState
from deck import Deck
from hand_eval import HandEvaluator

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
        elif num_players == 2:
            POS_NAMES[1 - player_idx - ((dealer_idx + 1) % 2)]
        else:
            return f"Pos{player_idx}"
        
    def run(self, num_hands=10):
        print(f"Starting poker game run with {len(self.players)} players for {num_hands} hands.")
        for i in range(num_hands):
            print(f"\n{'='*10} Hand {i+1}/{num_hands} {'='*10}")
            self._rotate_dealer()
            dealer_name = self.players[self.dealer_position].name
            dealer_pos_name = self.get_position_name(self.dealer_position, self.dealer_position, len(self.players))
            print(f"Dealer: {dealer_pos_name} ({dealer_name}, Index: {self.dealer_position})")

            self._play_hand()

            print("\nPlayer stacks after hand:")
            for p_idx, player in enumerate(self.players):
                try: stack_val = player.stack
                except AttributeError: stack_val = "N/A"
                pos_name = self.get_position_name(p_idx, self.dealer_position, len(self.players))
                print(f"  {pos_name} {p_idx} ({player.name}): {stack_val}")

            players_with_chips = [p for p in self.players if hasattr(p, 'stack') and p.stack > 0]
            if len(players_with_chips) <= 1:
                 print("\nGame over: Only one player (or fewer) has chips remaining.")
                 break