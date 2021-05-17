import numpy as np
# from agents.agent_random.random import generate_move_random
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, \
    PLAYER2, PlayerAction, initialize_game_state, PlayerAction, \
    PLAYER2_PRINT, PLAYER1_PRINT

# from agents.common import BoardPiece, SavedState, PlayerAction
from typing import Optional, Callable, Tuple


def test_generate_move():
    from agents.agent_minimax.minimax import generate_move
    from tests.test_boards import TestBoards
    i = 0
    while i < 10:
        i += 1
        board = initialize_game_state()
        player = np.random.choice([PLAYER1, PLAYER2])
        saved_state = {PLAYER1: None}
        ret, _ = generate_move(board, player, saved_state)

        assert isinstance(ret, PlayerAction)
        assert ret in np.arange(0, board.shape[1], 1)
        assert board[board.shape[0]-1, ret] == NO_PLAYER



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

def test_minimax():
    from agents.agent_minimax.minimax import generate_move, evaluate_min, \
        max_player_move, evaluate_max
    from tests.test_boards import TestBoards

    board = TestBoards.board_minimax_depth2
    print(TestBoards.minimax_depth2)
    alpha = -99999
    beta = 99999
    player = PLAYER1
    score, ret = max_player_move(board, player, 5, alpha, beta)
    print(score, ret)

    board = TestBoards.board_minimax_2
    print(TestBoards.minimax_2)
    score, ret = max_player_move(board, player, 5, alpha, beta)
    print(score, ret)

    board = TestBoards.board_minimax_3
    player = PLAYER2
    print(TestBoards.minimax_3)
    score, ret = max_player_move(board, player, 5, alpha, beta)
    print(score, ret)


def test_evaluate():
    from agents.agent_minimax.minimax import evaluate_min, evaluate_max
    from tests.test_boards import TestBoards
    board = TestBoards.board_evaluate_1
    print(TestBoards.evaluate_1)
    alpha = -99999
    beta = 99999
    player = PLAYER1
    print('Player: ', PLAYER1_PRINT)
    score_min = evaluate_min(board, player, alpha, beta)
    print(score_min)
    score_max = evaluate_max(board, player, alpha, beta)
    print(score_max)
    player = PLAYER2
    print('Player: ', PLAYER2_PRINT)
    score_min = evaluate_min(board, player, alpha, beta)
    print(score_min)
    score_max = evaluate_max(board, player, alpha, beta)
    print(score_max)


def test_heuristic():
    from agents.agent_minimax.minimax import MAX, MIN, heuristic
    from tests.test_boards import TestBoards
    board = TestBoards.board_heuristic_1
    ret = heuristic(board, PLAYER1, MAX, 2)
    print(TestBoards.heuristic_1)
    print('X, 2, max',ret)
    assert ret == 220

    ret = heuristic(board, PLAYER1, MIN, 2)
    print('X, 2, min', ret)
    assert ret == -220

    ret = heuristic(board, PLAYER2, MAX, 0)
    print('O, 0, max', ret)
    assert ret == 10

    ret = heuristic(board, PLAYER2, MAX, 4)
    print('O, 4, max', ret)
    assert ret == 20

    ret = heuristic(board, PLAYER1, MAX, 5)
    assert ret == 0

    board = TestBoards.board_heuristic_2
    ret = heuristic(board, PLAYER2, MIN, 3)
    assert ret == -10

    ret = heuristic(board, PLAYER2, MIN, 0)
    assert ret == 0



