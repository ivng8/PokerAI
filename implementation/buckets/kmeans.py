import os
import sys
import numpy as np
import random
import pickle
from collections import Counter
import time
from sklearn.cluster import KMeans

from hand_eval import HandEvaluator
from implementation.items.card import Card

class Clustering:
    NUM_PREFLOP_BUCKETS = 20
    NUM_FLOP_BUCKETS = 50
    NUM_TURN_BUCKETS = 50
    NUM_RIVER_BUCKETS = 50

    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    preflop_model_path = os.path.join(MODEL_DIR, 'preflop_kmeans_model.pkl')
    flop_model_path = os.path.join(MODEL_DIR, 'flop_kmeans_model.pkl')
    turn_model_path = os.path.join(MODEL_DIR, 'turn_kmeans_model.pkl')
    river_model_path = os.path.join(MODEL_DIR, 'river_kmeans_model.pkl')

    preflop_model = None
    flop_model = None
    turn_model = None
    river_model = None

    @classmethod
    def load_model(cls, model_attr_name, model_path):
        model = getattr(cls, model_attr_name)

        if model is not None:
            return model is not False

        file = open(model_path, 'rb')
        loaded_model = pickle.load(file)
        file.close()
    
        setattr(cls, model_attr_name, loaded_model)
        return True
    
    @classmethod
    def get_preflop_abstraction(cls, hole_cards):
        if not hole_cards or len(hole_cards) != 2:
            return cls.NUM_PREFLOP_BUCKETS - 1
        
        cls.load_model('preflop_model', cls.preflop_model_path)
        current_model = cls.preflop_model
        
        if isinstance(current_model, KMeans):
            features = cls.extract_preflop_features(hole_cards)
            cluster = current_model.predict(np.array(features).reshape(1, -1))[0]
            bucket = int(np.clip(cluster, 0, cls.NUM_PREFLOP_BUCKETS - 1))
            return bucket
        else:
            return cls.simple_preflop_bucket(hole_cards)
    
    @classmethod
    def get_postflop_abstraction(cls, hole_cards, community_cards):
        if community_cards:
            num_community = len(community_cards) 
        else:
            num_community = 0
        
        if num_community == 3:
            num_buckets = cls.NUM_FLOP_BUCKETS
            model_attr = 'flop_model'
            model_path = cls.flop_model_path
        elif num_community == 4:
            num_buckets = cls.NUM_TURN_BUCKETS
            model_attr = 'turn_model'
            model_path = cls.turn_model_path
        elif num_community == 5:
            num_buckets = cls.NUM_RIVER_BUCKETS
            model_attr = 'river_model'
            model_path = cls.river_model_path
        else:
            num_buckets = cls.NUM_FLOP_BUCKETS
            return num_buckets - 1

        # Basic input validation
        if not hole_cards or len(hole_cards) != 2 or not community_cards:
            return num_buckets - 1

        cls.load_model(model_attr, model_path)
        model = getattr(cls, model_attr)
        
        if isinstance(model, KMeans):
            features = cls.extract_postflop_features(hole_cards, community_cards)
            cluster = model.predict(np.array(features).reshape(1, -1))[0]
            bucket = int(np.clip(cluster, 0, num_buckets - 1))
            return bucket
        else:
            return cls.simple_postflop_bucket(hole_cards, community_cards, num_buckets)
        
    @classmethod
    def simple_preflop_bucket(cls, hole_cards):
        """ Simple deterministic preflop bucketing based on score heuristic. """
        if len(hole_cards) != 2:
            return cls.NUM_PREFLOP_BUCKETS - 1
        
        rank1, rank2 = sorted([c.rank for c in hole_cards], reverse=True)
        suited = hole_cards[0].suit == hole_cards[1].suit
        is_pair = rank1 == rank2
        gap = rank1 - rank2 if not is_pair else -1.0

        score = (rank1 + rank2) / 2.0
        if is_pair:
            score += 5.0
        if suited:
            score += 2.0
        if gap >= 0 and gap < 5:
            score += (4.0 - gap)  # Connector bonus

        min_score, max_score = 5.0, 21.0  # Approximate score range
        if max_score <= min_score:
            return 0  # Avoid division by zero

        normalized = (score - min_score) / (max_score - min_score)
        # Invert: High score -> Low bucket index
        bucket = cls.NUM_PREFLOP_BUCKETS - 1 - int(np.clip(normalized, 0, 1) * (cls.NUM_PREFLOP_BUCKETS - 1))
        return int(np.clip(bucket, 0, cls.NUM_PREFLOP_BUCKETS - 1))  # Ensure valid range
        
    @classmethod
    def simple_postflop_bucket(cls, hole_cards, community_cards, num_buckets):
        strength = cls.calculate_normalized_strength(hole_cards, community_cards)
        bucket = num_buckets - 1 - int(strength * (num_buckets - 1))
        return int(np.clip(bucket, 0, num_buckets - 1))
    
    @classmethod
    def extract_preflop_features(cls, hole_cards):
        rank1, rank2 = sorted([c.rank for c in hole_cards], reverse=True)
        suited = 1.0 if hole_cards[0].suit == hole_cards[1].suit else 0.0
        is_pair = 1.0 if rank1 == rank2 else 0.0
        gap = float(rank1 - rank2) if not is_pair else -1.0
        norm_rank1 = (rank1 - 2.0) / 12.0
        norm_rank2 = (rank2 - 2.0) / 12.0
        norm_gap = max(0.0, gap) / 12.0
        features = [norm_rank1, norm_rank2, suited, is_pair, norm_gap]
        return features
    
    @classmethod
    def extract_postflop_features(cls, hole_cards, community_cards, include_potential=False):
        if not hole_cards or len(hole_cards) != 2 or not community_cards or len(community_cards) < 3:
            return None
        
        hand_strength = cls.calculate_normalized_strength(hole_cards, community_cards)
        bp, bs, bc = cls.extract_board_features(community_cards)
        hp, htp, htr, hsd, hfd = cls.extract_hand_type_features(hole_cards, community_cards)
        
        features = [hand_strength, bp, bs, bc, float(hp), float(htp), float(htr), float(hsd), float(hfd)]
        
        return features
    
    @classmethod
    def calculate_normalized_strength(cls, hole_cards, community_cards):
        all_cards = hole_cards + community_cards
        if len(all_cards) < 5:
            return 0.0

        hand_eval_result = HandEvaluator.evaluate_hand(all_cards)
        hand_type_value = hand_eval_result[0]

        all_ranks = list(HandEvaluator.HAND_TYPES.values())
        min_r, max_r = min(all_ranks), max(all_ranks)
        
        if max_r <= min_r:
            return 0.5

        normalized = (hand_type_value - min_r) / float(max_r - min_r)
        return float(np.clip(normalized, 0.0, 1.0))
    
    @classmethod
    def extract_hand_type_features(cls, hole_cards, community_cards):
        """ Extract binary hand type/draw features. """
        all_cards = hole_cards + community_cards
        n = len(all_cards)
        if n < 5:
            return (0, 0, 0, 0, 0)

        hand_rank_val, _ = HandEvaluator.evaluate_hand(all_cards)

        h_p = 1 if hand_rank_val >= HandEvaluator.HAND_TYPES.get('pair', 1) else 0
        h_tp = 1 if hand_rank_val >= HandEvaluator.HAND_TYPES.get('two_pair', 2) else 0
        h_tr = 1 if hand_rank_val >= HandEvaluator.HAND_TYPES.get('three_of_a_kind', 3) else 0

        ranks = sorted([c.rank for c in all_cards], reverse=True)
        suits = [c.suit for c in all_cards]
        s_counts = Counter(suits)
        u_ranks = sorted(list(set(ranks)), reverse=True)

        h_fd = 1 if any(c == 4 for c in s_counts.values()) else 0
        h_sd = 0
        if len(u_ranks) >= 4:
            # Simplified OESD/Gutshot check
            for i in range(len(u_ranks) - 3):
                if u_ranks[i] - u_ranks[i+3] <= 4:  # Check if 4 ranks span 4 or 5 positions
                    h_sd = 1
                    break
            # Wheel draw check (A + 3 low cards)
            if not h_sd and 14 in u_ranks and len({r for r in u_ranks if r <= 5}) >= 3:
                h_sd = 1

        return h_p, h_tp, h_tr, h_sd, h_fd
    
    @classmethod
    def extract_board_features(cls, community_cards):
        """ Extract numerical board features (paired, suited, connected). Returns tuple or raises error. """
        n = len(community_cards)
        if n < 3: return (0.0, 0.0, 0.0)

        ranks = [c.rank for c in community_cards]; suits = [c.suit for c in community_cards]
        r_counts = Counter(ranks); s_counts = Counter(suits); u_ranks = sorted(list(set(ranks)))

        pair_f = 0.0; p_counts = [c for c in r_counts.values() if c>=2]
        if len(p_counts) >= 2 or any(c>=3 for c in p_counts): pair_f = 1.0
        elif len(p_counts)==1: pair_f = 0.5

        suit_f = 0.0; max_s_count = max(s_counts.values()) if s_counts else 0
        if max_s_count >= 5: suit_f = 1.0
        elif max_s_count == 4: suit_f = 2.0/3.0
        elif max_s_count == 3: suit_f = 1.0/3.0

        conn_f = 0.0
        if len(u_ranks) >= 3:
             is_v_conn, is_m_conn = False, False
             for i in range(len(u_ranks) - 2):
                 if u_ranks[i+2] - u_ranks[i] == 2: is_v_conn = True; break
             if not is_v_conn:
                 for i in range(len(u_ranks) - 2):
                      if u_ranks[i+2] - u_ranks[i] <= 3: is_m_conn = True; break
             # Wheel check (Ace + 2 low cards) implies medium connectivity if not already higher
             if not is_v_conn and not is_m_conn and 14 in u_ranks and len({r for r in u_ranks if r <= 5}) >= 2:
                 is_m_conn = True

             if is_v_conn: conn_f = 1.0
             elif is_m_conn: conn_f = 0.5

        return pair_f, suit_f, conn_f
    
    @staticmethod
    def calculate_hand_potential(hole_cards, community_cards, num_simulations=50):
        """ Estimates Ppot, Npot via Monte Carlo. """
        current_hand = hole_cards + community_cards
        n_curr = len(current_hand)
        if n_curr < 5:
            return 0.0, 0.0  # Cannot calculate potential pre-flop

        curr_rank = HandEvaluator.evaluate_hand(current_hand)

        used_set = frozenset(current_hand)
        deck_list = [Card(r, s) for r in range(2, 15) for s in Card.SUITS if Card(r,s) not in used_set]
        n_draw = max(0, 5 - len(community_cards))
        if n_draw == 0:
            return 0.0, 0.0  # No potential on river

        if len(deck_list) < 2 + n_draw:  # Need opp hole cards + runout cards
            return 0.0, 0.0  # Changed from raising an error to returning default values

        pp_wins, np_losses = 0, 0
        pp_count, np_count = 0, 0

        for _ in range(num_simulations):
            deck_samp = deck_list[:]
            random.shuffle(deck_samp)
            opp_hole = deck_samp[:2]
            runout = deck_samp[2 : 2+n_draw]
            final_comm = community_cards + runout

            player_final = hole_cards + final_comm
            opp_final = opp_hole + final_comm
            if len(player_final) < 5 or len(opp_final) < 5:
                continue

            player_final_rank = HandEvaluator.evaluate_hand(player_final)
            opp_final_rank = HandEvaluator.evaluate_hand(opp_final)

            opp_curr = opp_hole + community_cards
            opp_curr_rank = HandEvaluator.evaluate_hand(opp_curr) if len(opp_curr) >= 5 else (-1,[])

            is_ahead = curr_rank > opp_curr_rank
            is_behind = curr_rank < opp_curr_rank
            wins_sd = player_final_rank > opp_final_rank

            if is_ahead:
                np_count += 1
                np_losses += (0 if wins_sd else 1)
            elif is_behind:
                pp_count += 1
                pp_wins += (1 if wins_sd else 0)

        Ppot = (pp_wins / pp_count) if pp_count > 0 else 0.0
        Npot = (np_losses / np_count) if np_count > 0 else 0.0
        return float(Ppot), float(Npot)
    
    @staticmethod
    def train_models(num_samples_preflop=20000, num_samples_postflop=50000, random_state=42, include_potential_in_features=False):
        """
        Trains K-Means clustering models for each street and saves them.
        """
        # Create model directory if it doesn't exist
        os.makedirs(Clustering.MODEL_DIR, exist_ok=True)
        trained_models = {}
        
        # --- Preflop Model ---
        model_name = 'preflop'
        num_clusters = Clustering.NUM_PREFLOP_BUCKETS
        model_path = Clustering.preflop_model_path
        
        # Generate data and train model
        preflop_data = Clustering.generate_synthetic_preflop_data(num_samples_preflop)
        if preflop_data:
            preflop_kmeans = KMeans(
                n_clusters=num_clusters,
                random_state=random_state,
                n_init=10
            )
            
            preflop_kmeans.fit(np.array(preflop_data))
            with open(model_path, 'wb') as f:
                pickle.dump(preflop_kmeans, f)
            trained_models['preflop'] = preflop_kmeans
        
        # --- Postflop Models ---
        postflop_configs = [
            ("Flop", 3, Clustering.NUM_FLOP_BUCKETS, Clustering.flop_model_path),
            ("Turn", 4, Clustering.NUM_TURN_BUCKETS, Clustering.turn_model_path),
            ("River", 5, Clustering.NUM_RIVER_BUCKETS, Clustering.river_model_path)
        ]
        
        # Generate postflop data
        postflop_data = Clustering.generate_synthetic_postflop_data(5, num_samples_postflop, include_potential_in_features)
        
        if postflop_data:
            postflop_data_np = np.array(postflop_data)
            for round_name, num_comm, num_clusters, model_path in postflop_configs:
                kmeans = KMeans(
                    n_clusters=num_clusters,
                    random_state=random_state,
                    n_init=10
                )
                
                kmeans.fit(postflop_data_np)
                with open(model_path, 'wb') as f:
                    pickle.dump(kmeans, f)
                trained_models[round_name.lower()] = kmeans
        
        return trained_models
    
    @staticmethod
    def generate_synthetic_preflop_data(num_samples):
        """ Generate features for random preflop hands. """
        data = []
        count = 0
        attempts = 0
        all_cards = [Card(r,s) for r in range(2, 15) for s in Card.SUITS]
        max_attempts = num_samples * 5  # Allow more attempts

        while count < num_samples and attempts < max_attempts:
            attempts += 1
            c1, c2 = random.sample(all_cards, 2)
            features = Clustering.extract_preflop_features([c1, c2])
            if features is not None:  # Keep basic validation
                data.append(features)
                count += 1

        return data
    
    @staticmethod
    def generate_synthetic_postflop_data(num_community_cards, num_samples, include_potential):
        """ Generate features for random postflop hands/boards. """
        data = []
        count = 0
        all_cards = [Card(r,s) for r in range(2, 15) for s in Card.SUITS]
        cards_needed = 2 + num_community_cards
        max_attempts = num_samples * 10  # Allow more attempts

        if len(all_cards) < cards_needed:
            return []  # Not enough cards in deck

        for _ in range(max_attempts):
            if count >= num_samples:
                break
                
            sampled_cards = random.sample(all_cards, cards_needed)
            hole_cards = sampled_cards[:2]
            community = sampled_cards[2:]
            features = Clustering.extract_postflop_features(hole_cards, community, include_potential)
            
            if features is not None:  # Keep basic validation
                data.append(features)
                count += 1

        return data