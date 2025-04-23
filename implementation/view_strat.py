import pickle

pkl_path = 'models/6max_test_run/final_strategy_optimized.pkl'

with open(pkl_path, 'rb') as f:
    strategy = pickle.load(f)

with open('strategy.txt', 'w') as out_file:
    out_file.write(str(strategy))

print("Strategy dumped to strategy.txt")
