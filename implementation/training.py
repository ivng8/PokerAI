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
        
        # Track utility for progress monitoring
        utility_sum = 0
        
        for i in range(iterations):
            # Run one iteration of CFR
            utility = self.run_cfr()
            utility_sum += utility
            
            # Update statistics
            self.training_stats['iterations'] += 1
            
            # More detailed stats every 100 iterations
            if (i + 1) % 100 == 0 or i == iterations - 1:
                avg_utility = utility_sum / min(100, i+1)
                self.training_stats['avg_utility'].append(avg_utility)
                utility_sum = 0
            
            # Save checkpoint if needed
            if (i + 1) % checkpoint_interval == 0 or i == iterations - 1:
                self.save_checkpoint()
                exploitability = self.estimate_exploitability()
                self.training_stats['exploitability'].append(exploitability)
        
        # Update training time
        self.training_stats['training_time'] += time.time() - start_time
        
        # Save final model
        self.save_checkpoint(is_final=True)
        
        return self.training_stats
    
    def run_cfr(self):
        game = GameState(
            num_players=6, 
            starting_stack=self.starting_stack, 
            small_blind=self.small_blind, 
            big_blind=self.big_blind)
        
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
        """Estimate the exploitability of the current strategy using GameState directly."""
        # Create strategy object from current CFR strategy
        strategy = self.cfr_trainer.get_strategy()
        strategy_player = MCCFRStrategy()
        strategy_player.strategy = strategy
        
        total_utility = 0
        
        # Simulate hands
        for _ in range(num_hands):
            # Create game state for this simulation with 6 players
            game_state = GameState(
                num_players=6, 
                starting_stack=self.starting_stack, 
                small_blind=self.small_blind, 
                big_blind=self.big_blind
            )
            
            # Set up new hand with random dealer position
            dealer_position = random.randint(0, 5)  # 0-5 for 6 players
            initial_stacks = [float(self.starting_stack)] * 6
            game_state.start_new_hand(dealer_pos=dealer_position, player_stacks=initial_stacks)
            
            # Play until the hand is complete
            while not game_state.is_terminal():
                current_player_idx = game_state.current_player_idx
                
                # Skip if no valid player or player cannot act
                if current_player_idx < 0:
                    break
                    
                if current_player_idx < len(game_state.player_folded) and game_state.player_folded[current_player_idx]:
                    game_state.rotate_turn()
                    continue
                    
                if current_player_idx < len(game_state.player_all_in) and game_state.player_all_in[current_player_idx]:
                    game_state.rotate_turn()
                    continue
                
                # Get available actions
                available_actions = game_state.get_available_actions()
                if not available_actions:
                    game_state.rotate_turn()
                    continue
                
                # Determine action based on player
                if current_player_idx == 0:  # Strategy player
                    action = strategy_player.get_action(game_state, current_player_idx)
                else:  # Random exploiter
                    action = random.choice(available_actions)
                
                # Apply action and continue
                game_state = game_state.apply_action(action)
            
            # Calculate utility for strategy player
            utility = game_state.get_utility(0, initial_stacks)
            total_utility += utility
        
        # Calculate average utility normalized by big blinds
        avg_utility = total_utility / num_hands
        exploitability = avg_utility / self.big_blind
        
        return exploitability
    
    def save_checkpoint(self, is_final=False):
        strategy = self.cfr_trainer.get_strategy()
        
        # Determine file path based on checkpoint type
        if is_final:
            strategy_file = os.path.join(self.output_dir, "final_strategy.pkl")
        else:
            strategy_file = os.path.join(self.output_dir, f"strategy_iter_{self.training_stats['iterations']}.pkl")
        
        # Save strategy
        with open(strategy_file, 'wb') as f:
            pickle.dump(strategy, f)
        
        # Save training statistics
        stats_file = os.path.join(self.output_dir, "training_stats.pkl")
        with open(stats_file, 'wb') as f:
            pickle.dump(self.training_stats, f)

    def load_checkpoint(self, checkpoint_file):
        """Load a checkpoint to resume training."""
        # Load strategy
        with open(checkpoint_file, 'rb') as f:
            strategy = pickle.load(f)
        
        # Recreate information sets in the CFR trainer
        for key, strat in strategy.items():
            self.cfr_trainer.get_info_set(key, list(strat.keys()))
            
        # Load training statistics if available
        stats_file = os.path.join(self.output_dir, "training_stats.pkl")
        if os.path.exists(stats_file):
            with open(stats_file, 'rb') as f:
                self.training_stats = pickle.load(f)

    def get_trained_strategy(self):
        """Get the current trained strategy."""
        strategy_obj = MCCFRStrategy()
        strategy_obj.strategy = self.cfr_trainer.get_strategy()
        return strategy_obj