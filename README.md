# Texas Hold'em Poker Bot: A CFR/DLS Implementation

This repository contains a implementation of a No-Limit Hold'em (NLHE) poker bot designed for 6-max games. (Inspired by Pluribus)

## Key Features

*   **Counterfactual Regret Minimization (CFR):** Employs CFR variants for computing strategies
*   **Blueprint Strategy:** MCCFR training generates a comprehensive strategy profile covering numerous game states
*   **Depth-Limited Search :** Integrates search (inspired by Pluribus) to refine blueprint strategy decisions based on the exact game state encountered during play
*   **Buckets:** Utilizes abstraction techniques to group strategically similar hands, managing state and action space complexity
*   **Comprehensive Game Engine:** Includes a detailed game engine accurately modeling rules

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ivng8/PokerAI
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage (Command-Line Interface)
Train a blueprint strategy using CFR.

```bash
python main.py --mode train --iterations 5000 --output_dir ./models/optimized_6max --checkpoint_freq 500 --num_players 6 --optimized --num_workers 4
```
```bash
--iterations: Number of training iterations. More iterations generally yield stronger strategies but require more time.
--output_dir: Directory to save strategy checkpoints and the final strategy (final_strategy.pkl).
--checkpoint_freq: How often to save intermediate models.
--num_players: Set to 6 for 6-max training.
--optimized: Use the parallel OptimizedSelfPlayTrainer.
--num_workers: Number of CPU cores for parallel training.
```
