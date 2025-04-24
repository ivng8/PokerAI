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
python main.py --iterations 5000 --checkpoint_interval 500
```
```bash
