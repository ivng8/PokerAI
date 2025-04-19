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