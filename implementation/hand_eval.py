from collections import Counter
from itertools import combinations

class HandEvaluator:
    HAND_TYPES = {
        'high_card': 0,
        'pair': 1,
        'two_pair': 2,
        'three_of_a_kind': 3,
        'straight': 4,
        'flush': 5,
        'full_house': 6,
        'four_of_a_kind': 7,
        'straight_flush': 8,
        'royal_flush': 9
    }

    @staticmethod
    def evaluate_hand(cards):
        if len(cards) > 5:
            return HandEvaluator.find_best_hand(cards)
        
        sorted_cards = sorted(cards, key=lambda card: card.rank, reverse=True)
        
        is_flush = len(set(card.suit for card in cards)) == 1
        
        ranks = [card.rank for card in sorted_cards]
        if set(ranks) == {14, 5, 4, 3, 2}:
            is_straight = True
            ranks = [5, 4, 3, 2, 1]
        else:
            is_straight = (max(ranks) - min(ranks) == 4 and len(set(ranks)) == 5)
        
        rank_counts = Counter(ranks)
        count_values = sorted(rank_counts.values(), reverse=True)
        
        if is_straight and is_flush:
            if ranks[0] == 14 and ranks[1] == 13:  # A, K, Q, J, 10
                return (HandEvaluator.HAND_TYPES['royal_flush'], [])
            else:
                return (HandEvaluator.HAND_TYPES['straight_flush'], ranks)
        
        if count_values[0] == 4:
            quads_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
            kicker = [rank for rank, count in rank_counts.items() if count == 1][0]
            return (HandEvaluator.HAND_TYPES['four_of_a_kind'], [quads_rank, kicker])
        
        if count_values[0] == 3 and count_values[1] == 2:
            trips_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            return (HandEvaluator.HAND_TYPES['full_house'], [trips_rank, pair_rank])
        
        if is_flush:
            return (HandEvaluator.HAND_TYPES['flush'], ranks)
        
        if is_straight:
            return (HandEvaluator.HAND_TYPES['straight'], ranks)
        
        if count_values[0] == 3:
            trips_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            kickers = sorted([rank for rank, count in rank_counts.items() if count == 1], reverse=True)
            return (HandEvaluator.HAND_TYPES['three_of_a_kind'], [trips_rank] + kickers)
        
        if count_values[0] == 2 and count_values[1] == 2:
            pairs = sorted([rank for rank, count in rank_counts.items() if count == 2], reverse=True)
            kicker = [rank for rank, count in rank_counts.items() if count == 1][0]
            return (HandEvaluator.HAND_TYPES['two_pair'], pairs + [kicker])
        
        if count_values[0] == 2:
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            kickers = sorted([rank for rank, count in rank_counts.items() if count == 1], reverse=True)
            return (HandEvaluator.HAND_TYPES['pair'], [pair_rank] + kickers)
        
        return (HandEvaluator.HAND_TYPES['high_card'], ranks)
    
    @staticmethod
    def find_best_hand(cards):
        best_value = (-1, [])
        
        for hand in combinations(cards, 5):
            hand_value = HandEvaluator.evaluate_hand(list(hand))
            if hand_value > best_value:
                best_value = hand_value
        
        return best_value