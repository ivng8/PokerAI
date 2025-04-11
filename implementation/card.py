class Card:

    RANKS = {
        2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
        9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
    }

    SUITS = {
        'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'
    }

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def get_rank(self):
        return self.RANKS[self.rank]
    
    def get_suit(self):
        return self.SUITS[self.suit]
    
    def copy(self):
        return Card(self.rank, self.suit)
    
    def is_equal(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    def get_str(self):
        return f"{self.get_rank()}{self.get_suit()}"