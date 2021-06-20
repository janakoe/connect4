import numpy as np

from agents.common import NO_PLAYER, PLAYER1, \
    PLAYER2, initialize_game_state, PlayerAction, \
    PLAYER2_PRINT, PLAYER1_PRINT, valid_action

def test_init():
    from agents.agent_montecarlo import MonteCarloNode

    MonteCarloNode(parent, last_action, state, unexpanded_moves)

def test_UCB1():
    from agents.agent_montecarlo import MonteCarloNode
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    mcn = MonteCarloNode(None, board, player, None, valid_action(board))

    ucb1 = mcn.UCB1(np.sqrt(2))
    print(ucb1)
