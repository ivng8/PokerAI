import os
import argparse

from training import SelfTrainer

def main():
    parser = argparse.ArgumentParser(description='Train poker bot using self-play')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='../../models')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    trainer = SelfTrainer(output_dir, 50, 100, 10000)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(iterations=args.iterations, checkpoint_interval=args.checkpoint_interval)

if __name__ == "__main__":
    main()
