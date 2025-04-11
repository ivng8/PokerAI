from deck import Deck
from card import Card

class GameState:
    PREFLOP, FLOP, TURN, RIVER, SHOWDOWN, HAND_OVER = 0, 1, 2, 3, 4, 5
    ROUND_NAMES = {0:"Preflop", 1:"Flop", 2:"Turn", 3:"River", 4:"Showdown", 5:"Hand Over"}
    MAX_RAISES_PER_STREET = 7

    def __init__(self, num_players=6, starting_stack=10000, small_blind=50, big_blind=100):
        if not (2 <= num_players <= 9): raise ValueError("Num players must be 2-9")
        self.num_players = num_players
        self.small_blind = float(small_blind)
        self.big_blind = float(big_blind)
        self.player_stacks = [float(starting_stack)] * num_players
        self.hole_cards = [[] for _ in range(num_players)]
        self.player_total_bets_in_hand = [0.0] * num_players
        self.player_bets_in_round = [0.0] * num_players
        self.player_folded = [False] * num_players
        self.player_all_in = [False] * num_players
        self.active_players = list(range(num_players))
        self.community_cards = []
        self.pot = 0.0
        self.betting_round = self.PREFLOP
        self.deck = Deck()
        self.dealer_position = 0
        self.current_player_idx = -1
        self.current_bet = 0.0
        self.last_raiser = None
        self.last_raise = 0.0
        self.players_acted_this_round = set()
        self.verbose_debug = False
        self.raise_count_this_street = 0