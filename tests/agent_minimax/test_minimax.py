import numpy as np

from agents.common import NO_PLAYER, PLAYER1, \
    PLAYER2, initialize_game_state, PlayerAction, \
    PLAYER2_PRINT, PLAYER1_PRINT


def test_generate_move():
    """
    Test generate_move function of minimax agent of the following:
        - return is instance of PlayerAction
        - returned column is in range of board
        - returned column is not full
        - if only one valid column left - plays that column
    """
    from agents.agent_minimax.minimax import generate_move
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
        # check if random agent played only possible action j
        ret, _ = generate_move(board, player, saved_state)
        assert ret == j


def test_max_player_move():
    """
    Test function max_player_move by checking generated moves for the
    following example boards.
    """
    from agents.agent_minimax.minimax import max_player_move
    from tests.test_boards import TestBoards

    # assert if max agent plays 1 or 4 to prevent win of min agent in second
    # next move
    board = TestBoards.board_minimax_depth2
    print(TestBoards.minimax_depth2)
    alpha = -99999
    beta = 99999
    player = PLAYER1
    score, ret = max_player_move(board, player, 5, alpha, beta)
    print(score, ret)
    assert ret == 1 or ret == 4

    # assert if max agent plays 1 to prevent win of min agent in next move
    board = TestBoards.board_minimax_2
    print(TestBoards.minimax_2)
    score, ret = max_player_move(board, player, 5, alpha, beta)
    print(score, ret)
    assert ret == 1

    # assert if max agent plays 1 to win
    board = TestBoards.board_minimax_2
    player = PLAYER2
    print(TestBoards.minimax_2)
    score, ret = max_player_move(board, player, 5, alpha, beta)
    print(score, ret)
    assert ret == 1


def test_evaluate():
    """
    Test functions evaluate_min and evaluate max (-> called if depth=0) by
    checking score and column of selected move for the following example
    boards.
    """

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
    assert score_min[0] == -1*(2*100 + 3*10)
    score_max = evaluate_max(board, player, alpha, beta)
    print(score_max)
    assert score_max[0] == 2*100 + 3*10

    player = PLAYER2
    print('Player: ', PLAYER2_PRINT)

    score_min = evaluate_min(board, player, alpha, beta)
    print(score_min)
    assert score_min[0] == (100 + 10) * -1
    score_max = evaluate_max(board, player, alpha, beta)
    print(score_max)
    assert score_max[0] == 100 + 10


def test_heuristic():
    """
    Test the heuristic for different example boards
    """
    from agents.agent_minimax.minimax import MAX, MIN, heuristic
    from tests.test_boards import TestBoards

    board = TestBoards.board_heuristic_1
    print(TestBoards.heuristic_1)

    print('Player: ', PLAYER1_PRINT)
    print('last_action: ', 2)
    ret = heuristic(board, PLAYER1, MAX, 2)
    print('max', ret)
    assert ret == 220
    ret = heuristic(board, PLAYER1, MIN, 2)
    print('min', ret)
    assert ret == -220

    ret = heuristic(board, PLAYER1, MIN, 2)
    print('last_action: ', 0, 'min', ret)
    assert ret == -220

    print('Player: ', PLAYER2_PRINT)
    ret = heuristic(board, PLAYER2, MAX, 0)
    print('last_action: ', 0, 'max', ret)
    assert ret == 10

    ret = heuristic(board, PLAYER2, MAX, 4)
    print('last action: ', 4, 'max', ret)
    assert ret == 20

    board = TestBoards.board_heuristic_2
    ret = heuristic(board, PLAYER2, MIN, 3)
    print('last_action: ', 3, 'min', ret)
    assert ret == -10

    ret = heuristic(board, PLAYER2, MIN, 0)
    print('last_action: ', 0, 'min', ret)
    assert ret == 0
