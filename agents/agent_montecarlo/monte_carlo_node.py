import numpy as np
from agents.common import apply_player_action, change_player, valid_action, \
    PlayerAction, BoardPiece, connected_four


class MonteCarloNode(object):

    def __init__(self, parent, board: np.ndarray,
                 player: BoardPiece, last_action: PlayerAction):
        """

        Parameters
        ----------
        parent: MonteCarloNode
            parent of node
        board: np.ndarray
            game board belonging to node
        player: BoardPiece
            player who chooses action (child node of current node)
        last_action: PlayerAction
            action that led to this node
        """

        # game state attributes
        self.board = board
        self.player = player
        self.last_action = last_action

        # Monte Carlo attributes
        self.n_simulations = 0
        self.n_wins = 0
        self.is_won = False

        # Tree attributes
        self.parent = parent
        self.children = {}

        # dict for children nodes for all valid actions: {action: child_node}
        # all child nodes are set to None (unexpanded node)
        for move in valid_action(self.board):
            self.children[move] = None

    def expand(self, action: PlayerAction):
        """
        Expands the tree for the given action, e.g. creates a new
        MonteCarloNode for the child of the current node that belongs to the
        new action. Returns the new child node.

        Parameters
        ----------
        action : PlayerAction
            chosen best action

        Returns
        -------
        self.children[action]: MonteCarloNode
            expanded child node
        """

        if not self.is_won:
            new_board = apply_player_action(self.board,
                                            action,
                                            self.player,
                                            copy=True)
            self.children[action] = MonteCarloNode(self,
                                                   new_board,
                                                   change_player(self.player),
                                                   PlayerAction(action))

            if connected_four(new_board, self.player, action):

                self.children[action].is_won = True

        return self.children[action]

    def unexpanded_actions(self) -> np.ndarray:
        """
        Checks if a child node of the current node is still unexpanded
        (None) and if so appends the corresponding action to a list.

        Returns
        -------
        np.ndarray of unexpanded actions

        """

        child_nodes = np.array(list(self.children.values()))
        actions = np.array(list(self.children.keys()))

        return actions[np.argwhere(child_nodes == None)[:, 0]]

    def is_fully_expanded(self):
        """
        Returns whether current node cannot be further expanded, e.g all
        child nodes are not None.

        Returns
        -------
        boolean
        """

        if any(np.array(list(self.children.values())) == None):
            return False

        return True

    def UCB1(self, explore_param: float) -> float:
        """
        Function implements the UCB1 algorithm, it uses the numbers of wins
        and simulations of the children nodes, and the number of simulations
        of the parent node, to generate the UCB1 values for each child node
        according to:

            ucb1 = (wᵢ / sᵢ) + c * sqrt(ln(sₚ) / sᵢ )

            wᵢ : this node’s number of simulations that resulted in a win
                (self.n_wins)
            sᵢ : this node’s total number of simulations
                (self.n_simulations)
            sₚ : parent node’s total number of simulations
                (self.parent.n_simulations)
            c : exploration parameter
                (explore_param)

        Parameters
        ----------
        explore_param : float

        Returns
        -------
        ucb1: float

        """

        if (self.n_simulations != 0) & (self.parent.n_simulations != 0):
            exploitation = (self.n_wins / self.n_simulations)
            exploration = np.sqrt(np.log(self.parent.n_simulations) /
                                  self.n_simulations)
            return exploitation + explore_param * exploration
        else:
            return np.random.randint(-5, 5)
