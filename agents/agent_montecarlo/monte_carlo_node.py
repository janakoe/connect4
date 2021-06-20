import numpy as np
from agents.common import BoardPiece, SavedState, PlayerAction, \
    apply_player_action, connected_four, NO_PLAYER, PLAYER1, PLAYER2, \
    connected_n, change_player, valid_action
from typing import Tuple, Optional


class MonteCarloNode(object):

    #def __init__(self, parent: MonteCarloNode, last_action: PlayerAction,
                 #unexpanded_moves: object) -> object:

    def __init__(self, parent, board, player,
                 last_action):


        '''
        parent is the parent MonteCarloNode,
        last_action is the move made from the parent to get to this node,

        unexpandedPlays is an array of legal Plays that can be made from this node,
        children - dic
        '''
        # move made

        # game state
        self.board = board
        self.player = player
        self.last_action = last_action

        # Monte Carlo attributes
        self.n_simulations = 0
        self.n_wins = 0

        # Tree attributes
        self.parent = parent
        self.children = {}

        #for move in unexpanded_moves:
        for move in valid_action(self.board):
            #self.children[move.hash()] = {'last_action': move, 'node': None}
            self.children[move] = None


    #Expand the specified child last_action and return the new child node.
    def expand(self, action): # new_board???
        new_board = apply_player_action(self.board, action, self.player,
                                        copy=True)
        self.children[action] = MonteCarloNode(self,
                                               new_board,
                                               change_player(self.player),
                                               action)

        return MonteCarloNode

    def all_actions(self):
        ret = []
        for child in self.children:
            ret.append(child.play)
        return ret

    def unexpanded_actions(self):
        ret = []
        # TODO: optimization
        #ret = self.children.keys()[not self.children.values()]

        for child in self.children:
            if not self.children[child]:
                ret.append(child.play)
        return ret

    def is_fully_expanded(self):
        for child in self.children:
            if self.children[child] == None:
                return False
        return True

    # Whether this node is terminal in the game tree, NOT INCLUSIVE of termination due to winning.
    def is_leaf(self):
        if len(self.children) == 0:
            return True
        return False

    def UCB1(self, explore_param):
        if self.n_simulations != 0:
            exploitation = (self.n_wins / self.n_simulations)
            exploration = np.sqrt(np.log(self.parent.n_simulations)/
                              self.n_simulations)
            return exploitation + explore_param * exploration
        else:
            return np.random.randint(-5,5)




    # unnötig
    # Get the MonteCarloNode corresponding to the given last_action.
    def get_child_node(self, last_action):
        child = self.children[last_action]
        # child = self.children[last_action.hash()]?
        # if child == undefined
        #    raise error
        # elif child.node == None:
        #  raise ("Child is not expanded!")
        return child[last_action]
