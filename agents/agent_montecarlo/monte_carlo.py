import numpy as np
import time
from agents.common import BoardPiece, SavedState, PlayerAction, \
    apply_player_action, connected_four, change_player, valid_action, \
    connected_n


from typing import Tuple, Optional
from agents.agent_montecarlo.monte_carlo_node import MonteCarloNode


# Get the best move from available statistics.

def generate_move(board: np.ndarray,
                  player: BoardPiece,
                  saved_state):

    print('saved state beginning of montecarlo: ', saved_state)
    print('player1: ', saved_state[1])
    print('player2: ', saved_state[2])

    # build monte carlo search tree:
    if saved_state[player] is None:
        mcst = MonteCarlo(board, player, explore_param=np.sqrt(2))

    else:
        #print()
        action_opponent = saved_state[2]
        print('action_opponent: ', action_opponent)
        mcst = saved_state[player]
        print('saved state player1: - (mcst)', mcst)


        mcst.root = mcst.root.children[2]

    #mcst = MonteCarlo(board, player, explore_param=np.sqrt(2))
    mcst.run_search(timeout=1)
    action = mcst.best_action()

    mcst.root = mcst.root.children[action]
    print('mcst: ', mcst)
    #saved_state[player] = mcst

    #print('saved state at end of mcst:', saved_state)
    #saved_state[player].root = mcst.root.children[action]

    return PlayerAction(action), mcst


class MonteCarlo:

    def __init__(self, board, player, explore_param=np.sqrt(2)):
        self.explore_param = explore_param
        self.root = MonteCarloNode(None, board, player, None)

    # From given state, repeatedly run MCTS to build statistics.
    def run_search(self, timeout=1):
        # TODO: check for wins - in which methods??:

        end = time.time() + timeout
        while time.time() < end:

            # phase 1 - Selection: Select until not fully expanded OR leaf
            node = self.select(self.root)

            winner = node.player

            # TODO: try exept - add error raised for none in connect for
            if node.last_action is None:
                action = np.random.choice(node.unexpanded_actions())
                child = node.expand(action)
                # self.expand(node)
                # phase 3 - Simulation: Play game to terminal state, return
                # winner
                winner = self.simulate(child)

            elif not connected_four(node.board, node.player,
                                     node.last_action):
                # phase 2 - Expansion: Expand a random unexpanded child node
                action = np.random.choice(node.unexpanded_actions())
                node.expand(action)
                # self.expand(node)
                # phase 3 - Simulation: Play game to terminal state, return
                # winner
                winner = self.simulate(node)

            # phase 4 - Backpropagation: Update ancestor statistics
            self.backprop(node, winner)

    # phase 1 - Selection: Select until not fully expanded OR leaf
    def select(self, node):

        if not node.is_fully_expanded():
            return node

        selected_child = None
        maximum = np.NINF
        for action, child_node in node.children.items():
            ucb = child_node.UCB1(self.explore_param)
            if ucb > maximum:
                maximum = ucb
                selected_child = child_node
        return self.select(selected_child)

    # phase 2 - Expansion: Expand a random unexpanded child node
    # def expansion(self, node):
    #    action = np.random.choice(node.unexpanded_actions())
    #    node.expand(action)

    # phase 3 - Simulation: Play game to terminal state, return winner
    def simulate(self, node):

        board = node.board
        player = node.player
        last_action = node.last_action

        while not connected_four(board, player, last_action):
            try: action = np.random.choice(valid_action(board, player,
                                           last_action))
            except: return None

            board = apply_player_action(board, action, player, copy=True)
            player = change_player(player)

        return player

    # phase 4 - Backpropagation: Update ancestor statistics
    def backprop(self, node, winner):
        if node.parent is not None:
            node.n_simulations += 1
            if winner is not None:
                if node.player != winner:
                    node.n_wins += 1
            self.backprop(node.parent, winner)

    def best_action(self):

        # TODO: policy option: robust child vs highest win rate

        node = self.root
        maximum = np.NINF

        # TODO: check if all children are expanded, otherwise not enough
        #  information (MonteCarloNode.is_fully_expanded())
        if not node.is_fully_expanded():
            return 3
            print('not fully expanded')

        else:
            player_action = np.random.choice(valid_action(node.board,
                                                          node.player,
                                                          node.last_action))
            print(player_action)

            for action in node.children:
                child_node = node.children[action]
                print('action: ', action,'n wins: ', child_node.n_wins)

                if child_node.n_wins > maximum:
                    maximum = child_node.n_wins
                    player_action = PlayerAction(action)

        #for action in node.children:
        #    child_node = node.children[action]
        #    print('action: ', action,'n simulations: ',
        #         child_node.n_simulations)
        #    if child_node.n_simulations > maximum:
        #        maximum = child_node.n_simulations
        #        player_action = PlayerAction(action)
        # catch direct wins in this and next round:
            #board_new = apply_player_action(node.board, action,
            #                                node.player, copy=True)
            #if connected_four(board_new, node.player, action):
            #    return action
            #elif connected_n(board_new, node.player, action, 3):

        return player_action




# If given state does not exist, create dangling node.

    # def make_node(state):
    # '''if (!this.nodes.has(state.hash()))
    #      let unexpandedPlays = this.game.legalPlays(state).slice()
    #      let node = new MonteCarloNode(null, null, state, unexpandedPlays)
    #      this.nodes.set(state.hash(), node)'''
