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
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    return MonteCarlo(board, player, explore_param=np.sqrt(2))

def simulate_tree():
    import time
    mcst = create_empty_tree()
    timeout = 3
    end = time.time() + timeout

    while time.time() < end:
        node = mcst.select(mcst.root)
        action = np.random.choice(node.unexpanded_actions())
        child = node.expand(action)
        winner = mcst.simulate(child)
        mcst.backprop(node, winner)

    return mcst, child, winner

def create_ucb_node():
    root_ucb = create_root()
    root_ucb.n_simulations = 1
    for action in range(7):
        root_ucb.expand(action)
        n_wins = np.random.randint(0, 10)
        n_sim = np.random.randint(1, 5)
        root_ucb.children[action].n_wins = n_wins
        root_ucb.children[action].n_simulations = n_wins + n_sim
    return root_ucb

class TestTrees:
    """
    Generates example trees for the tests
    """

    #tree = create_empty_tree()

    #root = create_root()
    #root_ucb = create_root()

    #root_ucb.n_simulations = 1
    #for action in range(7):
    #    root_ucb.expand(action)
    #    n_wins = np.random.randint(0, 10)
    #    n_sim = np.random.randint(1, 5)
     #   root_ucb.children[action].n_wins = n_wins
     #    root_ucb.children[action].n_simulations = n_wins + n_sim

    #simulated_tree = simulate_tree()



