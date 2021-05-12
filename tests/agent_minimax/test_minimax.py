import numpy as np
# from agents.agent_random.random import generate_move_random
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, \
    PLAYER2, PlayerAction, initialize_game_state, PlayerAction

# from agents.common import BoardPiece, SavedState, PlayerAction
from typing import Optional, Callable, Tuple


def test_minimax():
    from agents.agent_minimax.minimax import generate_move

    i = 0
    while i < 100:
        i += 1
        board = initialize_game_state()
        player = np.random.choice([PLAYER1, PLAYER2])
        ret = generate_move(board, player)

        assert isinstance(ret, PlayerAction)
        assert ret in np.arange(0, board.shape[1], 1)
        assert board[board.shape[0]-1, ret] == NO_PLAYER

    # first : depth: 2

    # win if possible in move
    # test: do not do move so that other person wins in next move

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
        ret = generate_move(board, player)
        assert ret == j


def test_minimax_valid_moves():
    from agents.agent_minimax.minimax import generate_move
    from tests.test_boards import TestBoards
    i = 0

    while i < 50:
        i += 1
        # create full board with only 1 valid action in column j
        board = initialize_game_state()
        max_row, max_column = board.shape
        player = np.random.choice([PLAYER1, PLAYER2])
        j = np.random.randint(0, max_column)
        board[:] = player
        board[max_row-1, j] = NO_PLAYER
        saved_state = {PLAYER1: None}
        # check if random agent played chose only possible action j
        ret, _ = generate_move(board, player, saved_state)
        assert ret == j

    #board = TestBoards.board_valid_move
    #player = PLAYER2
    #saved_state = {PLAYER1: None}
    #ret, _ = generate_move(board, player, saved_state)
    #assert ret == 5


def test_heuristic_simple():
    from agents.agent_minimax.minimax import MAX, MIN, heuristic
    from tests.test_boards import TestBoards
    board = TestBoards.board_heuristic_1
    ret = heuristic(board, PLAYER2, MIN, 0)
    print(ret)
    #assert ret == 15

    board = TestBoards.board_heuristic_2
    ret = heuristic(board, PLAYER2, MIN, 3)
    #assert ret == -15
    print(ret)

    board = TestBoards.board_heuristic_3
    ret = heuristic(board, PLAYER1, MAX, 3)
    #assert ret == -15
    print(ret)

    board = TestBoards.board_heuristic_4
    ret = heuristic(board, PLAYER1, MAX, 3)
    #assert ret == -15
    print(ret)


def test_heuristic():
    from agents.agent_minimax.minimax import MAX, MIN, heuristic_2
    from tests.test_boards import TestBoards
    board = TestBoards.board_3_1
    ret = heuristic_2(board, PLAYER2, MIN, 5)
    ret2 = heuristic_2(board, PLAYER1, MAX, 0)
    #assert ret == 200
    print(ret)
    print(ret2)
