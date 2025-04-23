class Player:

    def __init__(self, name, stack=10000, position=0, human=False, random=False):
        self.name = name
        self.stack = stack
        self.position = position
        self.hole_cards = []
        self.is_active = True
        self.current_bet = 0
        self.total_bet = 0
        self.is_all_in = False
        self.human = human
        self.random = random
    
    def get_cards(self, cards):
        self.hole_cards = cards

    def bet(self, amount):
        actual_amount = min(amount, self.stack)
        
        self.stack -= actual_amount
        self.current_bet += actual_amount
        self.total_bet += actual_amount
        
        if self.stack == 0:
            self.is_all_in = True
        
        return actual_amount
    
    def fold(self):
        self.is_active = False

    def reset(self):
        self.hole_cards = []
        self.is_active = True
        self.current_bet = 0
        self.total_bet = 0
        self.is_all_in = False