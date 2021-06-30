import numpy as np
from agents.common import apply_player_action, change_player, valid_action, \
    PlayerAction, BoardPiece
#from agents.agent_montecarlo.monte_carlo import valid_action

class MonteCarloNode(object):

    # def __init__(self, parent: MonteCarloNode, last_action: PlayerAction,
    # unexpanded_moves: object) -> object:

    def __init__(self, parent, board: np.ndarray,
                 player: BoardPiece, last_action: PlayerAction):

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

        # for move in unexpanded_moves:
        for move in valid_action(self.board, self.player, self.last_action):
            # self.children[move.hash()] = {'last_action': move, 'node': None}
            self.children[move] = None

    # Expand the specified child last_action and return the new child node.
    def expand(self, action):

        new_board = apply_player_action(self.board,
                                        action,
                                        self.player,
                                        copy=True)
        self.children[action] = MonteCarloNode(self,
                                               new_board,
                                               change_player(self.player),
                                               action)

        return self.children[action]

    def unexpanded_actions(self):
        ret = []

        # TODO: optimization
        # ret = self.children.keys()[not self.children.values()]

        for child in self.children:
            if not self.children[child]:
                ret.append(child)
        return ret

    # TODO: optimization
             #if not all(node.children.values()):
             #    return node

    def is_fully_expanded(self):
        for child in self.children:
            if self.children[child] is None:
                return False
        return True

    def UCB1(self, explore_param):
        if self.n_simulations != 0 & self.parent.n_simulations != 0:
            exploitation = (self.n_wins / self.n_simulations)
            exploration = np.sqrt(np.log(self.parent.n_simulations) /
                                  self.n_simulations)
            return exploitation + explore_param * exploration
        else:
            return np.random.randint(-5, 5)





    # Whether this node is terminal in the game tree, check if INCLUSIVE
    # of termination due to winning because no children created
    # TODO: use instead of checking for win again in run_sim monte carlo?
    def is_leaf(self):
        if len(self.children) == 0:
            return True
        return False


    # unnötig
    # Get the MonteCarloNode corresponding to the given last_action.
    # def get_child_node(self, last_action):
    #    child = self.children[last_action]
        # child = self.children[last_action.hash()]?
        # if child == undefined
        #    raise error
        # elif child.node == None:
        #  raise ("Child is not expanded!")
    #    return child[last_action]



    # def all_actions(self):
    #         ret = []
    #         for child in self.children:
    #             ret.append(child.play)
    #         return ret
