import numpy as np
import time
from agents.common import BoardPiece, PlayerAction, apply_player_action, \
    connected_four, change_player, valid_action
from agents.agent_montecarlo.monte_carlo_node import MonteCarloNode


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
        whereby the value for the Monte Carlo player is type MonteCarlo and for
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
        mcst.root = mcst.root.children[action_opponent]

    mcst.run_search(timeout=2)
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

    def run_search(self, timeout: int = 1):
        """
        From given node, repeatedly run the Monte Carlo tree search to build
        statistics until timeout is reached.

        Parameters
        ----------
        timeout: int
            simulation time in seconds
        """

        # count = 0

        end = time.time() + timeout
        while time.time() < end:
            # count += 1

            # phase 1 - SELECTION:
            selected_node = self.select(self.root)

            # phase 2 - EXPANSION
            if not selected_node.is_won:

                action_expansion = np.random.choice(selected_node.unexpanded_actions())
                expanded_node = selected_node.expand(action_expansion)

                # phase 3 - SIMULATION:
                winner = self.simulate(expanded_node)

            else:
                expanded_node = selected_node
                winner = change_player(selected_node.player)

            # phase 4 - BACKPROPAGATION:
            self.backpropagation(expanded_node, winner)

        # print('number simulations: ', count)

    def select(self, node: MonteCarloNode):
        """
        Recursively select child node following the UCB1 algorithm until
        reached node is not fully expanded.

        Parameters
        ----------
        node: MonteCarloNode
            node from which to select a child node

        Returns
        -------
        selected child node: MonteCarloNode
        """

        if not node.is_fully_expanded():
            return node

        selected_child = node.children[np.random.choice(valid_action(
                         node.board))]
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
            node from which to select a child node

        Returns
        -------
        winner: BoardPiece
            Player who wins the game
        """

        board = node.board
        player = change_player(node.player)
        last_action = node.last_action

        # while game is not won
        while not connected_four(board, player, last_action):
            try:
                action = np.random.choice(valid_action(board))
            except:
                # simulated game ends drawn
                return None

            player = change_player(player)
            board = apply_player_action(board, action, player, copy=True)
            last_action = action

        return BoardPiece(player)

    def backpropagation(self, node: MonteCarloNode, winner: BoardPiece):
        """
        Back propagate  simulated win, e.g updates all ancestor statics
        starting from the simulation node by recursively calling the
        backpropagation function.

        Parameters
        ----------
        node: MonteCarloNode
            node for which the simulation was run - starting point of backpropagation
        winner: BoardPiece
        """

        node.n_simulations += 1
        if node.parent is not None:
            if winner is not None:
                # unequality because each node’s statistics are used for its
                # parent node’s choice, not its own
                if node.player != winner:
                    node.n_wins += 1
            self.backpropagation(node.parent, winner)

    def best_action(self, visualise=False) -> PlayerAction:
        """
        Chooses the best action for the current player according to the
        Monte Carlo Tree, e.g selects the child node that has the most number
        of wins or simulations depending on mode.

        Parameters
        ----------
        visualise:
            whether or not to print number of wins and number of simulations
            of all children

        Returns
        -------
        player_action: PlayerAction
            best column to play according to Monte Carlo Tree search
        """
        root = self.root
        maximum = np.NINF

        while not self.root.is_fully_expanded():
            # print('root was not fully expanded - extra simulation')
            self.run_search(timeout=1)

        for action in root.children:
            child_node = root.children[action]

            if visualise:
                print('action: ', action, 'n wins: ', child_node.n_wins)
                print('action: ', action, 'n sim: ', child_node.n_simulations)

            if child_node.n_wins > maximum:
                maximum = child_node.n_wins
                player_action = PlayerAction(action)

        return PlayerAction(player_action)
