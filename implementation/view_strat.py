import pickle

# Load the trained strategy
with open("path/to/models/final_strategy.pkl", "rb") as f:
    strategy_dict = pickle.load(f)

# Examine the strategy
print(f"Number of information sets: {len(strategy_dict)}")