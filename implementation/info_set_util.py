from implementation.game_state import GameState
from implementation.buckets.kmeans import Clustering

def generate_key(game_state, player_idx):
    cards_part = "NOCARDS"
    pos_part = "POS_ERR"
    hist_part = "BH_ERR"

    hole = []
    if game_state.hole_cards:
        hole = game_state.hole_cards[player_idx]
    num_comm = len(game_state.community_cards)

    if len(hole) == 2:
        if num_comm == 0:
            bucket = Clustering.get_preflop_abstraction(hole)
            cards_part = f"PRE_ENH{bucket}"
        else:
            round_names = {3: "FLOP", 4: "TURN", 5: "RIVER"}
            round_prefix = round_names.get(num_comm, f"POST{num_comm}")
            
            bucket = Clustering.get_postflop_abstraction(hole, game_state.community_cards)
            cards_part = f"{round_prefix}_ENH{bucket}"

    elif hole:
        hole_str = "_".join(sorted(str(c) for c in hole))
        if game_state.community_cards:
            comm_str = "_".join(sorted(str(c) for c in game_state.community_cards))
        else:
            comm_str = "noc"
        cards_part = f"RAW_{hole_str}_{comm_str}"

    position_relative = game_state.get_position(player_idx)
    pos_part = f"POS{position_relative}"
    
    hist_part = game_state.get_betting_history()
    if not hist_part and game_state.betting_round == GameState.PREFLOP:
        if not game_state.players_acted_this_round:
            hist_part = "start"
        else:
            hist_part = "empty_hist?"
    elif not hist_part:
        hist_part = "postflop_start?"
        hist_part = "BH_ERR_Empty"

    final_key = f"{str(cards_part)}|{str(pos_part)}|{str(hist_part)}"
    return final_key