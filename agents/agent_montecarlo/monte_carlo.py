import numpy as np
import time
import json
from agents.common import BoardPiece, SavedState, PlayerAction, \
    apply_player_action, connected_four, NO_PLAYER, PLAYER1, PLAYER2, \
    connected_n, valid_action, change_player


from typing import Tuple, Optional
from agents.agent_montecarlo import MonteCarloNode

 # Get the best move from available statistics.
def generate_move(board: np.ndarray,
                  player: BoardPiece,
                  saved_state: Optional[SavedState])\
                    -> Tuple[PlayerAction, Optional[SavedState]]:

    # build search tree:
    mcts = MonteCarlo(...)

    mcts.run_search(board, player, timeout=3)
    # stats = mcts.getStats(state)
    # play = mcts.bestPlay(state, "robust")

    action = mcts.best_move(board, player, )
    return action
    # return PlayerAction(column), saved_state


class MonteCarlo:

    def __init__(self, board, player, explore_param=np.sqrt(2)):
        self.explore_param = explore_param
        self.root = MonteCarloNode(None, board, player, None, valid_action(board))


    # From given state, repeatedly run MCTS to build statistics.
    def run_search(self, board, player, timeout=3):

        # self.make_node(state)

        end = time.time() + timeout * 1000
        while time.time() < end:
            node = self.select(self.root)

            # TODO: check for win:

            if not node.is_leaf() and gamestate:
            #if connected_four(board_new, min_player, PlayerAction(column)):
                node = self.expand(node)
                winner = self.simulate(node)

            backprop(node, winner)


    # If given state does not exist, create dangling node.

    #def make_node(state):
        '''if (!this.nodes.has(state.hash()))
          let unexpandedPlays = this.game.legalPlays(state).slice()
          let node = new MonteCarloNode(null, null, state, unexpandedPlays)
          this.nodes.set(state.hash(), node)'''


    # 4 phases of Monte Carlo Tree Search

    # phase 1 - Selection: Select until not fully expanded OR leaf
    def select(self, node):

        if not all(node.children.values()):
            return node

        child_node = None
        maximum = np.NINF
        for action, child_node in node.children.items():
            ucb = child_node.UCB1(self.explore_param)
            if ucb > maximum:
                maximum = ucb

        return self.select(child_node)


    # phase 2 - Expansion: Expand a random unexpanded child node
    def expand(self, node):
        action = np.random.choice(node.unexpanded_actions())
        node.expand(action)

    # phase 3 - Simulation: Play game to terminal state, return winner
    def simulate(self, node):
        board = node.board
        player = node.player
        last_action = node.last_action

        while not connected_four(board, player, last_action):
            action = np.random.choice(valid_action(node.board))
            board = apply_player_action(board, action, player, copy=True)
            player = change_player(player)

        return player

    # phase 4 - Backpropagation: Update ancestor statistics
    def backprop(self, node, winner):
        if node.parent is not None:
            node.n_simulations += 1
            if node.player != winner:
                node.n_wins += 1
            self.backprop(node.parent, winner)



    # wohin?
    def best_move(self, board):
        # TODO: policy option: robust child vs highest win rate

        #why??
        #self.make_node(state)


        node = self.nodes[json.dumps(board)]

        maximum = np.NINF

        # TODO: check if all children are expanded, otherwise not enough
        #  information (MonteCarloNode.is_fully_expanded())
        player_action = np.random.choice(get_legal_moves(board))
        for action in node.childreen:
            child_node = node.childreen[action]
            if child_node.n_simulations > maximum:
                maximum = child_node.n_simulations
                player_action = PlayerAction(action)



