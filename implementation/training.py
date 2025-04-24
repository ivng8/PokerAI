import os
import pickle
import random
import time
import numpy as np

from implementation.game_state import GameState
from implementation.strategy import MCCFRStrategy
from implementation.trainer import Trainer

class SelfTrainer:

    def __init__(self, output_dir, small_blind=50, big_blind=100, starting_stack=10000):
        self.output_dir = output_dir
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stack = starting_stack
        os.makedirs(output_dir, exist_ok=True)
        self.cfr_trainer = Trainer(GameState)
        
        self.training_stats = {
            'iterations': 0,
            'avg_utility': [],
            'exploitability': [],
            'training_time': 0
        }

    def train(self, iterations, checkpoint_interval=1000):
        start_time = time.time()
        utility_sum = 0
        
        for i in range(iterations):
            utility = self.run_cfr()
            utility_sum += utility
            self.training_stats['iterations'] += 1
            
            if (i + 1) % 100 == 0 or i == iterations - 1:
                avg_utility = utility_sum / min(100, i+1)
                self.training_stats['avg_utility'].append(avg_utility)
                utility_sum = 0
            
            if (i + 1) % checkpoint_interval == 0 or i == iterations - 1:
                self.save_checkpoint()
                exploitability = self.estimate_exploitability()
                self.training_stats['exploitability'].append(exploitability)
        
        self.training_stats['training_time'] += time.time() - start_time
        self.save_checkpoint(is_final=True)
        
        return self.training_stats
    
    def run_cfr(self):
        game = GameState(6, self.starting_stack, self.small_blind, self.big_blind)
        
        dealer_position = random.randint(0, 5)
        initial_stacks = [float(self.starting_stack)] * 6
        game.start_new_hand(dealer_position, initial_stacks)
        
        utility = self.cfr_trainer.calculate_cfr(
            game_state=game,
            reach_probs=np.ones(6, dtype=float),
            player_idx=0,
            initial_stacks=initial_stacks,
            weight=1.0,
            depth=0,
        )
        
        return utility
    
    def estimate_exploitability(self, num_hands=1000):
        strategy = self.cfr_trainer.get_strategy()
        strategy_player = MCCFRStrategy()
        strategy_player.strategy = strategy
        
        total_utility = 0
        
        for _ in range(num_hands):
            game_state = GameState(6, self.starting_stack, self.small_blind, self.big_blind)
            dealer_position = random.randint(0, 5)
            initial_stacks = [float(self.starting_stack)] * 6
            game_state.start_new_hand(dealer_position, initial_stacks)
            
            while not game_state.is_terminal():
                current_player_idx = game_state.current_player_idx
                if current_player_idx < 0:
                    break
                    
                if current_player_idx < len(game_state.player_folded) and game_state.player_folded[current_player_idx]:
                    game_state.rotate_turn()
                    continue
                    
                if current_player_idx < len(game_state.player_all_in) and game_state.player_all_in[current_player_idx]:
                    game_state.rotate_turn()
                    continue
                
                available_actions = game_state.get_available_actions()
                if not available_actions:
                    game_state.rotate_turn()
                    continue
                
                if current_player_idx == 0:
                    action = strategy_player.get_action(game_state, current_player_idx)
                else:
                    action = random.choice(available_actions)
                
                game_state = game_state.apply_action(action)
            
            utility = game_state.get_utility(0, initial_stacks)
            total_utility += utility
        
        avg_utility = total_utility / num_hands
        exploitability = avg_utility / self.big_blind
        
        return exploitability
    
    def save_checkpoint(self, is_final=False):
        strategy = self.cfr_trainer.get_strategy()
        
        if is_final:
            strategy_file = os.path.join(self.output_dir, "final_strategy.pkl")
        else:
            strategy_file = os.path.join(self.output_dir, f"strategy_iter_{self.training_stats['iterations']}.pkl")
        
        with open(strategy_file, 'wb') as f:
            pickle.dump(strategy, f)
        
        stats_file = os.path.join(self.output_dir, "training_stats.pkl")
        with open(stats_file, 'wb') as f:
            pickle.dump(self.training_stats, f)

    def load_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            strategy = pickle.load(f)
        
        for key, strat in strategy.items():
            self.cfr_trainer.get_info_set(key, list(strat.keys()))
            
        stats_file = os.path.join(self.output_dir, "training_stats.pkl")
        if os.path.exists(stats_file):
            with open(stats_file, 'rb') as f:
                self.training_stats = pickle.load(f)

    def get_trained_strategy(self):
        strategy_obj = MCCFRStrategy()
        strategy_obj.strategy = self.cfr_trainer.get_strategy()
        return strategy_obj