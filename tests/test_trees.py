from agents.agent_montecarlo.monte_carlo_node import MonteCarloNode
from agents.agent_montecarlo.monte_carlo import MonteCarlo
from agents.common import initialize_game_state, PLAYER1, PLAYER2
import numpy as np


def create_root():
    """
    Returns
    -------
    root node of initialized game state
    """
    parent = None
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    last_action = None

    return MonteCarloNode(parent, board, player, last_action)

def create_empty_tree():
    """
    Returns
    -------
    mcst object belonging to empty tree
    """
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    return MonteCarlo(board, player, explore_param=np.sqrt(2))

def simulate_tree():
    """
    Returns
    -------
    mcst object for which tree search was run for 3 s.
    """
    import time
    mcst = create_empty_tree()
    timeout = 3
    end = time.time() + timeout

    while time.time() < end:
        node = mcst.select(mcst.root)
        action = np.random.choice(node.unexpanded_actions())
        child = node.expand(action)
        winner = mcst.simulate(child)
        mcst.backpropagation(node, winner)

    return mcst, child, winner

def create_ucb_node():
    """
    Returns
    -------
    expended root node, whereby n_wins and n_simulations of every child node
    are assigned randomly to test ucb function
    """
    root_ucb = create_root()
    root_ucb.n_simulations = 1
    for action in range(7):
        root_ucb.expand(action)
        n_wins = np.random.randint(0, 10)
        n_sim = np.random.randint(1, 5)
        root_ucb.children[action].n_wins = n_wins
        root_ucb.children[action].n_simulations = n_wins + n_sim
    return root_ucb




