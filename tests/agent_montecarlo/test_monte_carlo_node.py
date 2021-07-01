import numpy as np

from agents.common import NO_PLAYER, PLAYER1, \
    PLAYER2, initialize_game_state, PlayerAction, \
    PLAYER2_PRINT, PLAYER1_PRINT, valid_action
from tests.test_trees import TestTrees
import tests.test_trees as tt

def test_init():
    """
    Test __init()__ function of MonteCarloNode class by checking if all class
    attributes are correctly initialized.
    """
    from agents.agent_montecarlo.monte_carlo_node import MonteCarloNode

    parent = None
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    last_action = None
    ret = MonteCarloNode(parent, board, player, last_action)

    assert isinstance(ret, MonteCarloNode)
    assert np.all(ret.board == board)
    assert ret.player == player
    assert ret.last_action == last_action
    assert ret.n_simulations == 0
    assert ret.n_wins == 0
    assert ret.parent is None
    assert len(ret.children) == 7

    # check if all childnodes are None
    for key in ret.children:
        assert not ret.children[key]


def test_expand():
    """
    Test expand() function of MonteCarloNode by checking if correct
    child node is expanded and type of MonteCarloNode
    """

    from agents.agent_montecarlo.monte_carlo_node import MonteCarloNode
    root = tt.create_root()
    action = np.random.randint(0, 7)
    assert not isinstance(root.children[action], MonteCarloNode)
    root.expand(action)
    assert isinstance(root.children[action], MonteCarloNode)


def test_unexpanded_actions():
    """
    Checks if ...
        ... unexpanded_actions() returns list of all actions for root node
            without expanded children
        ... after expanding a random childnode the corresponding action is
            not included in the returned list
    """

    root = tt.create_root()
    ret = root.unexpanded_actions()
    assert isinstance(ret, np.ndarray)
    assert len(ret) == 7

    action = np.random.randint(0, 7)
    root.expand(action)
    ret = root.unexpanded_actions()
    assert len(ret) == 6
    assert action not in ret


def test_is_fully_expanded():
    """
    Checks if ...
        ... is_fully_expanded() returns false for root node without expanded
            children
        ... after expanding all childnodes of the root node is_fully_expanded()
            returns true
    """
    from agents.agent_montecarlo.monte_carlo_node import MonteCarloNode
    root = tt.create_root()
    ret = root.is_fully_expanded()
    assert isinstance(ret, bool)
    assert not ret

    for action in range(7):
        root.expand(action)

    ret = root.is_fully_expanded()
    assert ret


def test_UCB1():
    """
    Tests implementation of UCB1 algorithm for children of node created in
    TestTrees.
    """

    node = tt.create_ucb_node()

    for action in range(7):
        c = np.sqrt(2)
        ret = node.children[action].UCB1(c)
        assert isinstance(ret, float)

        w_i = node.children[action].n_wins
        s_i = node.children[action].n_simulations
        s_p = node.n_simulations
        print(f'{action}. n_wins: ', node.children[action].n_wins)
        print(f'{action}. n_simulations: ', node.children[
                                            action].n_simulations)
        print(f'{action}. parent.n_simulations: ', node.n_simulations)
        print(f'{action}. ucb1: ', ret)
        assert ret == ((w_i / s_i) + c * np.sqrt(np.log(s_p) / s_i ))

