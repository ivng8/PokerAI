import random

from implementation.card import Card

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
    
    def clone(self):
        new_deck = Deck()
        new_deck.cards = [card.clone() for card in self.cards]
        return new_deck