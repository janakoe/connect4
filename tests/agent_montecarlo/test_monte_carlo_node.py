import numpy as np

from agents.common import NO_PLAYER, PLAYER1, \
    PLAYER2, initialize_game_state, PlayerAction, \
    PLAYER2_PRINT, PLAYER1_PRINT, valid_action

def test_init():
    from agents.agent_montecarlo.monte_carlo_node import MonteCarloNode

    parent = None
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    last_action = None
    node = MonteCarloNode(parent, board, player, last_action)

    assert np.all(node.board == board)
    assert node.player == player
    assert node.last_action == last_action
    assert node.n_simulations == 0
    assert node.n_wins == 0
    assert node.parent is None

    for key in node.children:
        assert not node.children[key]
    node.children[0] = node

    assert len(node.children) == 7



def test_UCB1():
    from agents.agent_montecarlo.monte_carlo_node import MonteCarloNode
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    mcn = MonteCarloNode(None, board, player, None)

    ucb1 = mcn.UCB1(np.sqrt(2))
    print(ucb1)


#def test_expand():
# expand(self, action):

#def unexpanded_actions(self):

#def is_fully_expanded(self):



# def is_leaf(self):
