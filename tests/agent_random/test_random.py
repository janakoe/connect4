import numpy as np
# from agents.agent_random.random import generate_move_random
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, \
    PLAYER2, PlayerAction, initialize_game_state, PlayerAction

# from agents.common import BoardPiece, SavedState, PlayerAction
from typing import Optional, Callable, Tuple


def test_generate_move_random():
    from agents.agent_random.random import generate_move_random

    i = 0
    while i < 100:
        i += 1
        board = initialize_game_state()
        player = np.random.choice([PLAYER1, PLAYER2])
        ret = generate_move_random(board, player)

        assert isinstance(ret, PlayerAction)
        assert ret in np.arange(0, board.shape[1], 1)
        assert board[board.shape[0]-1, ret] == NO_PLAYER


    i = 0
    while i < 100:
        i += 1
        # create full board with only 1 valid action in column j
        board = initialize_game_state()
        player = np.random.choice([PLAYER1, PLAYER2])
        j = np.random.randint(0, board.shape[1])
        board = player
        board[board.shape[0]-1, j] = NO_PLAYER

        # check if random agent played chose only possible action j
        ret = generate_move_random(board, player)
        assert ret == j
