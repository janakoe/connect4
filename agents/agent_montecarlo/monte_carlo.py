import numpy as np
import time
from agents.common import BoardPiece, SavedState, PlayerAction, \
    apply_player_action, connected_four, change_player, valid_action, \
    connected_n, GameState, check_end_state

from agents.agent_montecarlo.monte_carlo_node import MonteCarloNode


#TODO: bias towards left - shuffle before

def generate_move(board: np.ndarray,
                  player: BoardPiece,
                  saved_state: dict):
    """
    Generates move for player by calling running the tree search by calling
    run_search() and then selecting the best action by calling best_action().

    Parameters
    ----------
    board: np.ndarray
        current game board
    player: BoardPiece
        player for whom a move is generated
    saved_state: dict
        saved states for both players in form: {PLAYER1: ..., PLAYER2: ...}
        whereby the value for montecarlo player is type MonteCarlo and for
        the other player is int.

    Returns
    -------
    action: PlayerAction
        column to be played
    mcst: MonteCarlo
        MonteCarlo object with root node at the selected action

    """

    # create new monte carlo search tree:
    if saved_state[player] is None:
        mcst = MonteCarlo(board, player, explore_param=np.sqrt(2))

    # move root of monte carlo tree search to current node
    else:
        action_opponent = saved_state[change_player(player)]
        mcst = saved_state[player]
        print(mcst)
        mcst.root = mcst.root.children[action_opponent]

    # TODO: function direct wins/prevent wins:

    mcst.run_search(timeout=3)
    action = mcst.best_action()
    mcst.root = mcst.root.children[action]

    return PlayerAction(action), mcst


class MonteCarlo:

    def __init__(self, board: np.ndarray, player: BoardPiece,
                 explore_param: int = np.sqrt(2)):
        """
        Parameters
        ----------
        board: np.ndarray
            current game board
        player: BoardPiece
            current player
        explore_param: float
            explore parameter for UCB1 algorithm
        """
        self.explore_param = explore_param
        self.root = MonteCarloNode(None, board, player, None)

        #self.selected_node
        #self.sim_node = self.root
        #self.winner = None


    def run_search(self, timeout: int = 1):
        """
        From given node, repeatedly run the Monte Carlo tree search to build
        statistics until timeout is reached.

        Parameters
        ----------
        timeout: int
            simulation time in seconds

        """

        end = time.time() + timeout
        while time.time() < end:

            # phase 1 - SELECTION:
            selected_node = self.select(self.root)

            # phase 2 - EXPANSION
            # TODO: expand direct wins?
            action_expansion = np.random.choice(selected_node.
                                                unexpanded_actions())
            new_node = selected_node.expand(action_expansion)

            # phase 3 - SIMULATION:
            winner = self.simulate(new_node)

            # phase 4 - BACKPROPAGATION:
            self.backprop(new_node, winner)

    def select(self, node: MonteCarloNode):
        """
        Recursively select childnode following the UCB1 algorithm until
        reached node is not fully expanded.

        Parameters
        ----------
        node: MonteCarloNode
            node from which to select a childnode

        Returns
        -------
        childnode: MonteCarloNode
            selected childnode

        """

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

    def simulate(self, node: MonteCarloNode) -> BoardPiece:
        """
        Simulates game till terminal state by selecting actions randomly and
        return winner.

        Parameters
        ----------
        node: MonteCarloNode
            node from which to select a childnode

        Returns
        -------
        winner: BoardPiece
            Player who wins the game
        """

        board = node.board
        player = node.player
        last_action = node.last_action

        # while game is not won
        while not connected_four(board, change_player(player), last_action):
            try: action = np.random.choice(valid_action(board))
            except: return None

            board = apply_player_action(board, action, player, copy=True)
            player = change_player(player)
            last_action = action

        return BoardPiece(change_player(player))

    def backprop(self, node: MonteCarloNode, winner: BoardPiece):
        """
        Backpropagates simulated win, e.g updates all ancestor statics
        starting from the simulation node by recursivly calling the backprop
        function.

        Parameters
        ----------
        node: MonteCarloNode
            node for which the simulation was run - starting point of backprop
        winner: BoardPiece
        """
        if node.parent is not None:
            node.n_simulations += 1
            if winner is not None:
                # unequality because each node’s statistics are used for its
                # parent node’s choice, not its own
                if node.player != winner:
                    node.n_wins += 1
            self.backprop(node.parent, winner)

    def best_action(self, mode: str = 'n_wins') -> PlayerAction:
        """
        Chooses the best action for the current player according to the
        Monte Carlo Tree, e.g selects the childnode that has the most number
        of wins or simulations depending on mode.

        Parameters
        ----------
        n_mode: str
            'n_wins': selects child with most wins
            'n_simulations': selects child that was simulated the most

        Returns
        -------
        player_action: PlayerAction
            best column to play according to Monte Carlo Tree search
        """
        node = self.root
        maximum = np.NINF

        if not node.is_fully_expanded():
            print('not fully expanded')
            return 3

        if mode == 'n_wins':
            for action in node.children:
                child_node = node.children[action]
                print('action: ', action,'n wins: ', child_node.n_wins)
                if child_node.n_wins > maximum:
                    maximum = child_node.n_wins
                    player_action = PlayerAction(action)

        elif mode == 'n_sim':
            for action in node.children:
                child_node = node.children[action]
                print('action: ', action,'n sim: ', child_node.n_simulations)
                if child_node.n_simulations > maximum:
                        maximum = child_node.n_simulations
                        player_action = PlayerAction(action)

        return PlayerAction(player_action)

# TODO: check if all children are expanded, otherwise not enough
        #  information (MonteCarloNode.is_fully_expanded())
        #if not node.is_fully_expanded():
        #    print('not fully expanded')
        #    return 3
#board_new = apply_player_action(node.board, action,
            #                                node.player, copy=True)
            #if connected_four(board_new, node.player, action):
            #    return action
            #elif connected_n(board_new, node.player, action, 3):

