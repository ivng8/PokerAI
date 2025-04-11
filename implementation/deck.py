import random

from card import Card

class Deck:

    def __init__(self):
        self.cards = []
        self.reset()

    def reset(self):
        self.cards = []
        for suit in ['h', 'd', 'c', 's']:
            for rank in range(2, 15):
                self.cards.append(Card(rank, suit))

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop()
    
    def len(self):
        return len(self.cards)
    
    def get_str(self):
        return f"Deck({len(self.cards)} cards)"