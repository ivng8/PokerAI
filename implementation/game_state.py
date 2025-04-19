import numpy as np
from deck import Deck
from hand_eval import HandEvaluator
from copy import deepcopy

class GameState:
    PREFLOP, FLOP, TURN, RIVER, SHOWDOWN, HAND_OVER = 0, 1, 2, 3, 4, 5
    ROUND_NAMES = {0:"Preflop", 1:"Flop", 2:"Turn", 3:"River", 4:"Showdown", 5:"Hand Over"}
    MAX_RAISES_PER_STREET = 7

    def __init__(self, num_players=6, starting_stack=10000, small_blind=50, big_blind=100):
        self.num_players = num_players
        self.small_blind = float(small_blind)
        self.big_blind = float(big_blind)
        self.player_stacks = [float(starting_stack)] * num_players
        self.hole_cards = [[] for _ in range(num_players)]
        self.player_total_bets_in_hand = [0.0] * num_players
        self.player_bets_in_round = [0.0] * num_players
        self.player_folded = [False] * num_players
        self.player_all_in = [False] * num_players
        self.active_players = list(range(num_players))
        self.community_cards = []
        self.pot = 0.0
        self.betting_round = self.PREFLOP
        self.deck = Deck()
        self.dealer_position = 0
        self.current_player_idx = -1
        self.current_bet = 0.0
        self.last_raiser = None
        self.last_raise = 0.0
        self.players_acted_this_round = set()
        self.raise_count_this_street = 0
        self.action_sequence = []
        self.verbose_debug = False
        
    def next_active_player(self, start_idx):
        if not self.active_players: 
            return None
        valid_start = -1
        if 0 <= start_idx < self.num_players:
            valid_start = start_idx
        current_idx = (valid_start + 1) % self.num_players
        search_start_idx = current_idx

        for _ in range(self.num_players * 2):
            if current_idx in self.active_players and self.player_stacks[current_idx] > 0.01 and not self.player_folded[current_idx]:
                 return current_idx

            current_idx = (current_idx + 1) % self.num_players
            if current_idx == search_start_idx:
                break

        return None

    def offset_from_dealer(self, offset):
        if not self.active_players: 
            return None
        start_idx = (self.dealer_position + offset) % self.num_players
        current_idx = start_idx
        search_start_idx = current_idx

        for _ in range(self.num_players * 2):
            if current_idx in self.active_players and self.player_stacks[current_idx] > 0.01:
                return current_idx

            current_idx = (current_idx + 1) % self.num_players
            if current_idx == search_start_idx:
                break

        return None

    def start_new_hand(self, dealer_pos, player_stacks):
        self.hole_cards = [[] for _ in range(self.num_players)]
        self.community_cards = []
        self.pot = 0.0
        self.betting_round = self.PREFLOP
        self.player_bets_in_round = [0.0] * self.num_players
        self.player_total_bets_in_hand = [0.0] * self.num_players
        self.player_folded = [False] * self.num_players
        self.player_all_in = [False] * self.num_players
        self.current_player_idx = -1
        self.current_bet = 0.0
        self.last_raiser = None
        self.last_raise = 0.0
        self.players_acted_this_round = set()
        self.action_sequence = []
        self.raise_count_this_street = 0
        self.dealer_position = dealer_pos % self.num_players
        self.deck = Deck()
        self.deck.shuffle()
        self.player_stacks = [float(s) for s in player_stacks]
        self.active_players = [i for i, s in enumerate(self.player_stacks) if s > 0.01]

        if len(self.active_players) > 1:
            self.deal_hole()
            if self.betting_round == self.HAND_OVER:
                return
            self.put_blinds()
            if self.betting_round == self.HAND_OVER:
                return
            self.start_bets()
        else:
            self.betting_round = self.HAND_OVER
            self.current_player_idx = -1

    def deal_hole(self):
        if len(self.active_players) < 2:
            self.betting_round = self.HAND_OVER
            return

        start_player = self.offset_from_dealer(1)
        current_deal_idx = start_player
        for _ in range(2):
            for _ in range(len(self.num_players)):
                if current_deal_idx in self.active_players:
                    self.hole_cards[current_deal_idx].append(self.deck.deal())

                current_deal_idx = (current_deal_idx + 1) % self.num_players

    def deduct_bet(self, player_idx, amount_to_deduct):
        actual_deduction = min(amount_to_deduct, self.player_stacks[player_idx])
        if actual_deduction < 0.01:
            return 0.0

        self.player_stacks[player_idx] -= actual_deduction
        self.player_bets_in_round[player_idx] += actual_deduction
        self.player_total_bets_in_hand[player_idx] += actual_deduction
        self.pot += actual_deduction

        if abs(self.player_stacks[player_idx]) < 0.01:
            self.player_all_in[player_idx] = True

        return actual_deduction

    def put_blinds(self):
        if len(self.active_players) < 2:
            self.betting_round = self.HAND_OVER
            return

        sb_player, bb_player = None, None
        if len(self.active_players) == 2:
            sb_player = self.offset_from_dealer(0)
            bb_player = self.offset_from_dealer(1)
            if sb_player is None or bb_player is None:
                self.betting_round = self.HAND_OVER
                return
        else:
            sb_player = self.offset_from_dealer(1)
            bb_player = self.offset_from_dealer(2)
            if sb_player is None or bb_player is None:
                self.betting_round = self.HAND_OVER
                return
            if sb_player == bb_player:
                self.betting_round = self.HAND_OVER
                return

        self.raise_count_this_street = 0
        sb_amount = min(self.small_blind, self.player_stacks[sb_player])
        sb_posted_amount = self.deduct_bet(sb_player, sb_amount)
        if sb_posted_amount > 0.01:
            self.action_sequence.append(f"P{sb_player}:sb{int(round(sb_posted_amount))}")

        bb_amount_to_post = min(self.big_blind, self.player_stacks[bb_player])
        bb_posted_amount = self.deduct_bet(bb_player, bb_amount_to_post)
        if bb_posted_amount > 0.01:
            log_bb_amt = self.player_bets_in_round[bb_player]
            self.action_sequence.append(f"P{bb_player}:bb{int(round(log_bb_amt))}")

        self.current_bet = self.big_blind
        self.last_raise = self.big_blind
        if bb_posted_amount >= self.big_blind - 0.01:
            self.last_raiser = bb_player
            self.raise_count_this_street = 1
        elif sb_posted_amount > 0.01:
            self.last_raiser = sb_player
            self.current_bet = sb_posted_amount
            self.last_raise = sb_posted_amount
            self.raise_count_this_street = 1
        else:
            self.last_raiser = None
            self.current_bet = 0.0
            self.last_raise = self.big_blind
            self.raise_count_this_street = 0

    def start_bets(self):
        if self.betting_round != self.PREFLOP:
            self.current_bet = 0.0
            self.last_raiser = None
            self.last_raise = self.big_blind
            self.raise_count_this_street = 0
            self.player_bets_in_round = [0.0] * self.num_players

        self.players_acted_this_round = set()
        first_player_to_act = None

        if self.betting_round == self.PREFLOP:
            if len(self.active_players) == 2:
                first_player_to_act = self.offset_from_dealer(0)
            else:
                bb_player = self.offset_from_dealer(2)
                first_player_to_act = self.next_active_player(bb_player if bb_player is not None else self.dealer_position)
        else:
            first_player_to_act = self.next_active_player(self.dealer_position)
        if first_player_to_act is None:
            self.current_player_idx = -1
        else:
            self.current_player_idx = first_player_to_act
        self.current_player_idx = first_player_to_act if first_player_to_act is not None else -1

        if self.check_all_shoved():
            self.current_player_idx = -1


    def deal_board(self, burn=True):
        if burn:
            self.deck.deal()
        self.community_cards.append(self.deck.deal())

    def deal_flop(self):
        self.deck.deal()
        for _ in range(3):
            self.deal_board(False)
        self.betting_round = self.FLOP
        self.start_bets()

    def deal_turn(self):
        self.deal_board(True)
        self.betting_round = self.TURN
        self.start_bets()

    def deal_river(self):
        self.deal_board(True)
        self.betting_round = self.RIVER 
        self.start_bets()

    def check_all_shoved(self):
        non_folded_players = [p for p in range(self.num_players) if not self.player_folded[p]]

        if len(non_folded_players) < 2:
            return True

        still_act = 0
        for p_idx in non_folded_players:
            if not self.player_all_in[p_idx]:
                still_act += 1

        return still_act < 2

    def rotate_turn(self):
        if self.current_player_idx != -1:
            next_player = self.next_active_player(self.current_player_idx)
            if next_player is not None:
                self.current_player_idx = next_player  
            else:
                self.current_player_idx = -1

    def apply_action(self, action):
        action_type, amount_input = action
        amount = float(amount_input)
        
        acting_player_idx = self.current_player_idx
        if self.player_all_in[acting_player_idx]:
            new_state_skip = self.clone()
            new_state_skip.rotate_turn()
            if new_state_skip.betting_done():
                new_state_skip.move_round()
            return new_state_skip

        new_state = self.clone()
        new_state.action_logic(acting_player_idx, action_type, amount)

        if new_state.betting_done():
            new_state.move_round()
        else:
            new_state.rotate_turn()
            if new_state.current_player_idx != -1 and new_state.betting_done():
                new_state.move_round()

        return new_state


    def action_logic(self, p_idx, action_type, amount):
        player_stack = self.player_stacks[p_idx]
        current_round_bet = self.player_bets_in_round[p_idx]
        self.players_acted_this_round.add(p_idx)
        action_log_repr = f"P{p_idx}:"

        if action_type == "fold":
            self.player_folded[p_idx] = True
            if p_idx in self.active_players:
                self.active_players.remove(p_idx)
            action_log_repr += "f"
            if len([p for p in range(self.num_players) if not self.player_folded[p]]) <= 1:
                self.betting_round = self.HAND_OVER
                self.current_player_idx = -1

        elif action_type == "check":
            if self.current_bet - current_round_bet > 0.01:
                raise ValueError(f"Invalid check P{p_idx}: Bet={self.current_bet}, HasBet={current_round_bet}")
            action_log_repr += "k"

        elif action_type == "call":
            amount_needed = self.current_bet - current_round_bet
            if amount_needed <= 0.01:
                action_log_repr += "c0"
            else:
                call_cost = min(amount_needed, player_stack)
                if call_cost < 0: call_cost = 0
                self.deduct_bet(p_idx, call_cost)
                action_log_repr += f"c{int(round(self.player_bets_in_round[p_idx]))}"

        elif action_type == "bet":
            actual_bet_cost = min(amount, player_stack)
            is_all_in = abs(actual_bet_cost - player_stack) < 0.01
            
            self.deduct_bet(p_idx, actual_bet_cost)
            action_log_repr += f"b{int(round(actual_bet_cost))}"

            new_total_bet_level = self.player_bets_in_round[p_idx]
            self.current_bet = new_total_bet_level
            self.last_raise = new_total_bet_level
            self.last_raiser = p_idx
            self.raise_count_this_street = 1
            self.players_acted_this_round = {p_idx}
            if is_all_in:
                self.player_all_in[p_idx] = True

        elif action_type == "raise":
            total_bet_target = amount
            cost_to_reach_target = total_bet_target - current_round_bet
            actual_raise_cost = min(cost_to_reach_target, player_stack)
            actual_total_bet_reached = current_round_bet + actual_raise_cost
            is_all_in = abs(actual_raise_cost - player_stack) < 0.01

            self.deduct_bet(p_idx, actual_raise_cost)
            action_log_repr += f"r{int(round(actual_total_bet_reached))}"

            new_bet_level = actual_total_bet_reached
            self.last_raise = new_bet_level - self.current_bet
            self.current_bet = new_bet_level
            self.last_raiser = p_idx
            self.raise_count_this_street += 1
            self.players_acted_this_round = {p_idx}
            if is_all_in:
                self.player_all_in[p_idx] = True

        if len(action_log_repr) > len(f"P{p_idx}:"):
            self.action_sequence.append(action_log_repr)


    def get_betting_history(self):
        return ";".join(self.action_sequence)

    def get_available_actions(self):
        actions = []
        player_idx = self.current_player_idx

        if player_idx == -1:
            return actions

        if self.player_folded[player_idx] or self.player_all_in[player_idx] or self.player_stacks[player_idx] < 0.01:
            return actions

        player_stack = self.player_stacks[player_idx]
        current_round_bet = self.player_bets_in_round[player_idx]

        actions.append(("fold", 0))

        amount_to_call = self.current_bet - current_round_bet

        if amount_to_call < 0.01:
            actions.append(("check", 0))
        else:
            call_cost = min(amount_to_call, player_stack)
            actions.append(("call", int(round(call_cost))))

        if self.raise_count_this_street < self.MAX_RAISES_PER_STREET and player_stack > amount_to_call + 0.01:
            max_bet = current_round_bet + player_stack

            if self.current_bet < 0.01:
                action_prefix = "bet"
                bet_cost = min(player_stack, self.big_blind)
                min_bet = current_round_bet + bet_cost
            else:
                action_prefix = "raise"
                min_raise = max(self.last_raise, self.big_blind)
                new_bet = self.current_bet + min_raise
                min_bet = min(max_bet, new_bet)

            is_raise = (min_bet > self.current_bet + 0.01)

            if is_raise:
                actions.append((action_prefix, int(round(min_bet))))

            is_all_in_aggro = (max_bet > self.current_bet + 0.01)
            is_all_in_forced = abs(max_bet - min_bet) > 0.01

            if is_all_in_aggro and (not is_raise or is_all_in_forced) :
                 actions.append((action_prefix, int(round(max_bet))))

        return actions


    def betting_done(self):
        vpip = [p for p in range(self.num_players) if not self.player_folded[p]]
        if len(vpip) < 2:
            return True

        side_pot = []
        for p_idx in vpip:
            if not self.player_all_in[p_idx]:
                side_pot.append(p_idx)

        if len(side_pot) == 0:
            return True

        if len(side_pot) == 1:
            player = side_pot[0]

            is_preflop = self.betting_round == self.PREFLOP
            bb_player_idx = None
            if len(self.active_players) == 2:
                bb_player_idx = self.offset_from_dealer(1)
            elif len(self.active_players) > 2: 
                bb_player_idx = self.offset_from_dealer(2)

            has_acted = player in self.players_acted_this_round
            facing_bet = self.current_bet - self.player_bets_in_round[player] > 0.01
            is_bb_player = player == bb_player_idx
            no_reraise_yet = self.last_raiser == bb_player_idx

            if is_preflop and is_bb_player and no_reraise_yet and not facing_bet and not has_acted:
                return False

            return not facing_bet or has_acted

        for p_idx in side_pot:
            if abs(self.player_bets_in_round[p_idx] - self.current_bet) > 0.01:
                return False
            if p_idx not in self.players_acted_this_round:
                return False

        return True


    def move_round(self):
        vpip = [p for p in range(self.num_players) if not self.player_folded[p]]
        if len(vpip) < 2:
            self.betting_round = self.HAND_OVER
            self.current_player_idx = -1
            self.players_acted_this_round = set()
            return

        if self.check_all_shoved() and self.betting_round < self.SHOWDOWN:
            if self.betting_round == self.PREFLOP:
                self.deal_flop()
            if self.betting_round == self.FLOP:
                self.deal_turn()
            if self.betting_round == self.TURN:
                self.deal_river()
            if self.betting_round != self.HAND_OVER:
                self.betting_round = self.SHOWDOWN
                self.current_player_idx = -1
                self.players_acted_this_round = set()
            return

        if self.betting_round == self.PREFLOP:
            self.deal_flop()
        elif self.betting_round == self.FLOP:
            self.deal_turn()
        elif self.betting_round == self.TURN:
            self.deal_river()
        elif self.betting_round == self.RIVER:
            self.betting_round = self.SHOWDOWN
            self.current_player_idx = -1
            self.players_acted_this_round

    def is_terminal(self):
        vpip_count = len([p for p in range(self.num_players) if not self.player_folded[p]])
        if vpip_count <= 2:
            return True
        if self.betting_round >= self.SHOWDOWN:
             return True
        return False


    def get_utility(self, player_idx, initial_stacks=None):
        if not self.is_terminal():
            return 0.0
        
        i_s = initial_stacks[player_idx]
        initial_stack = float(i_s)

        current_copy = self.clone()
        c_s = current_copy.player_stacks[player_idx]
        current_stack = float(c_s)

        try:
            _ = current_copy.determine_winners()
            final_effective_stack = current_copy.player_stacks[player_idx]
        except:
            final_effective_stack = current_stack    
            
        utility = final_effective_stack - initial_stack

        if np.isnan(utility) or np.isinf(utility):
            utility = 0.0

        return utility


    def determine_winners(self, player_names=None):
        if not self.is_terminal():
            return []

        if self.pot < 0.01:
            self.pot = 0.0
            return []

        total_pot_to_distribute = self.pot
        self.pot = 0.0
        pots_summary = []

        vpip = [p for p in range(self.num_players) if not self.player_folded[p]]
        
        if not vpip:
            return []
        
        if len(vpip) == 1:
            winner_idx = vpip[0]
            amount_won = total_pot_to_distribute
            self.player_stacks[winner_idx] += amount_won
            pots_summary = [{'winners': [winner_idx], 'amount': amount_won, 'eligible': [winner_idx], 'desc': 'Uncontested'}]
            return pots_summary

        evaluated_hands = {}
        for p_idx in vpip:
            all_cards_for_eval = self.hole_cards[p_idx] + self.community_cards
            evaluated_hands[p_idx] = HandEvaluator.evaluate_hand(all_cards_for_eval)

        contributions = sorted([(p, self.player_total_bets_in_hand[p]) for p in vpip], key=lambda x: x[1])

        side_pots = []
        last_contribution_level = 0.0

        for p_idx_sp, total_contribution in contributions:
            contribution_increment = total_contribution - last_contribution_level
            if contribution_increment > 0.01:
                pot_amount = contribution_increment * len(vpip)
                if pot_amount > 0.01:
                    side_pots.append({'amount': pot_amount, 'eligible': vpip[:]})
                last_contribution_level = total_contribution

            if p_idx_sp in vpip:
                vpip.remove(p_idx_sp)

        if not side_pots and total_pot_to_distribute > 0.01:
             side_pots.append({'amount': total_pot_to_distribute, 'eligible': vpip[:]})
        
        distributed_total = 0
        for i, pot_info in enumerate(side_pots):
            pot_amount = pot_info.get('amount', 0.0)
            eligible_players_this_pot = pot_info.get('eligible', [])

            if pot_amount < 0.01 or not eligible_players_this_pot:
                 continue

            eligible_hands = {p: evaluated_hands[p] for p in eligible_players_this_pot if p in evaluated_hands}
            if not eligible_hands:
                 continue

            best_hand_value = max(eligible_hands.values())
            pot_winners = [p for p, hand_val in eligible_hands.items() if hand_val == best_hand_value]

            if pot_winners:
                 winner_share = pot_amount / len(pot_winners)
                 distributed_total += pot_amount # Track total distributed
                 for w_idx in pot_winners:
                     # Check bounds before adding winnings
                     if 0 <= w_idx < len(self.player_stacks):
                          self.player_stacks[w_idx] += winner_share

                 # Create summary entry for this pot
                 pot_desc = f"Side Pot {i+1}" if len(side_pots) > 1 else "Main Pot"
                 pots_summary.append({'winners':pot_winners, 'amount':pot_amount, 'eligible':eligible_players_this_pot, 'desc': pot_desc})

        return pots_summary

    def clone(self):
        return deepcopy(self)