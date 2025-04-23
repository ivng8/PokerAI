from collections import defaultdict

class InfoSet:

    def __init__(self, actions):
        unique_actions = []
        seen_actions = set()
        if not isinstance(actions, list): actions = []

        for action in actions:
            if isinstance(action, str):
                action_tuple = (action, 0)
            elif isinstance(action, tuple) and len(action) == 2:
                action_tuple = action
            else: 
                continue

            if action_tuple not in seen_actions:
                unique_actions.append(action_tuple)
                seen_actions.add(action_tuple)

        self.actions = unique_actions
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        for action in self.actions:
            self.regret_sum[action] = 0.0
            self.strategy_sum[action] = 0.0

    def get_strategy(self):
        if not self.actions:
            return {}

        strategy = {}
        normalization_sum = 0.0
        positive_regrets = {}

        for action in self.actions:
            regret = self.regret_sum.get(action, 0.0)
            positive_regret = max(0.0, regret)
            positive_regrets[action] = positive_regret
            normalization_sum += positive_regret

        if normalization_sum > 0:
            for action in self.actions:
                strategy[action] = positive_regrets[action] / normalization_sum
        else:
            num_actions = len(self.actions)
            if num_actions > 0:
                prob = 1.0 / num_actions
                for action in self.actions:
                    strategy[action] = prob
        
        return strategy
    
    def update_strategy_sum(self, current_strategy, weighted_reach_prob):
        for action in self.actions:
            action_prob = current_strategy.get(action, 0.0)
            self.strategy_sum[action] += weighted_reach_prob * action_prob

    def get_average_strategy(self):
        if not self.actions:
             return {}

        avg_strategy = {}
        normalization_sum = sum(self.strategy_sum.values())

        if normalization_sum > 0:
            for action in self.actions:
                action_sum = self.strategy_sum.get(action, 0.0)
                avg_strategy[action] = action_sum / normalization_sum
        else:
            num_actions = len(self.actions)
            if num_actions > 0:
                prob = 1.0 / num_actions
                for action in self.actions:
                    avg_strategy[action] = prob

        return avg_strategy