import os
import argparse
from training import SelfTrainer

def main():
    """Main function to run self-play training."""
    parser = argparse.ArgumentParser(description='Train poker bot using self-play')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of training iterations')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='How often to save checkpoints')
    parser.add_argument('--output_dir', type=str, default='../../models',
                        help='Directory to save training outputs')
    parser.add_argument('--small_blind', type=int, default=50,
                        help='Small blind amount')
    parser.add_argument('--big_blind', type=int, default=100,
                        help='Big blind amount')
    parser.add_argument('--starting_stack', type=int, default=10000,
                        help='Starting chip stack for each player')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
    
    # Create absolute path for output directory
    output_dir = os.path.abspath(args.output_dir)
    
    # Initialize trainer
    trainer = SelfTrainer(
        output_dir=output_dir,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_stack=args.starting_stack
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training
    trainer.train(
        iterations=args.iterations,
        checkpoint_interval=args.checkpoint_interval
    )

if __name__ == "__main__":
    main()
