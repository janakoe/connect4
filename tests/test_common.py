import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, \
    PLAYER2, PlayerAction
from tests.test_boards import TestBoards


def test_initialize_game_state():
    """
    Tests function initialize_game_state by checking if returned board is
    the right data type, shape and empty
    """
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
    """
    Tests function pretty print board by comparing returned String to
    desired String for hard coded examples.
    """
    from agents.common import pretty_print_board, initialize_game_state

    board = initialize_game_state()
    ret = pretty_print_board(board)

    assert isinstance(ret, str)
    assert ret == '|=============|\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'

    board[0, 0] = PLAYER1
    ret = pretty_print_board(board)
    assert ret == '|=============|\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|X            |\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'
    board[0, 2] = PLAYER2
    ret = pretty_print_board(board)
    assert ret == '|=============|\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|X   O        |\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'


def test_apply_player_action():
    """
    Test function apply_player_action by checking if it
        - changes the right position in the array
        - doesn´t change original board for copy_True
        - raises an error if the column is full
    """
    from agents.common import apply_player_action, initialize_game_state, \
        PLAYER1

    board = initialize_game_state()
    player = PLAYER1
    action = 2
    ret = apply_player_action(board, action, player)
    assert isinstance(ret, np.ndarray)
    assert ret[0, action] == 1

    # test if origin board unchanged with copy=True
    board = initialize_game_state()
    board_copy = np.copy(board)
    new_board = apply_player_action(board, action, player, True)
    assert np.all(board == board_copy)
    assert new_board[0, action] == 1

    # test if raises error for full board
    board = initialize_game_state()
    board += PLAYER1
    try:
        apply_player_action(board, action, player, True)
    except:
        assert True
    else:
        assert False


def test_string_to_board():
    """
    Tests function string_to_board by generating board, generating pretty
    string board and then calling string_to_board and asserting of returned
    board is equal the original board
    """
    from agents.common import string_to_board, initialize_game_state,\
        pretty_print_board

    board = initialize_game_state()
    number_pieces = 10
    i = np.random.randint(0, len(board), size=number_pieces)
    j = np.random.randint(0, len(board[0]), size=number_pieces)

    for n in range(number_pieces):
        player = np.random.choice([PLAYER1, PLAYER2])
        board[i[n], j[n]] = player
    print(board)

    board_str = pretty_print_board(board)
    print(board_str)
    res = string_to_board(board_str)
    print(res)
    assert np.all(res == board)


def test_connected_four():
    """
    Tests connected_four function by checking if it
        - recognises a random vertical, horizontal and diagonal win
        - evaluates two examples of not wins to false
    """
    from agents.common import connected_four, initialize_game_state

    # vertical
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    i = np.random.randint(0, len(board))
    j = np.random.randint(0, 4)
    board[i, np.arange(j, j+4)] = player
    print(board)
    assert connected_four(board, player, PlayerAction(j))

    # horizontal
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    i = np.random.randint(0, 3)
    j = np.random.randint(0, len(board[0]))
    board[np.arange(i, i+4), j] = player
    assert connected_four(board, player, PlayerAction(j))

    # diagonal
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    i = np.random.randint(0, 3)
    j = np.random.randint(0, 4)
    board[np.arange(i, i+4), np.arange(j, j+4)] = player
    print(board)
    assert connected_four(board, player, PlayerAction(j+3))

    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    i = np.random.randint(3, 6)
    j = np.random.randint(0, 4)
    board[np.arange(i, i-4, -1), np.arange(j, j+4)] = player
    print(board)
    assert connected_four(board, player, PlayerAction(j))

    # two examples for false
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    i = np.random.randint(3, 6)
    j = np.random.randint(0, 4)
    board[np.arange(i, i-3, -1), np.arange(j, j+3)] = player
    print(board)
    assert not connected_four(board, player, PlayerAction(j))

    board = TestBoards.board_test_4
    player = PLAYER1
    i = 1
    assert not connected_four(board, player, PlayerAction(i))


def test_connected_n():
    """
    Tests connected_n function by checking the number of connected n´s for
    example boards from the TestBoards class.
    """

    from agents.common import connected_n, initialize_game_state
    n = 3

    board = TestBoards.board_3_1
    print(board)
    assert connected_n(board, PLAYER1, PlayerAction(0), n) == 1
    assert connected_n(board, PLAYER2, PlayerAction(3), n) == 2
    assert connected_n(board, PLAYER2, PlayerAction(5), n) == 2

    board = TestBoards.board_3_2
    print(board)
    assert connected_n(board, PLAYER2, PlayerAction(5), n) == 0
    assert connected_n(board, PLAYER1, PlayerAction(1), n) == 0
    assert connected_n(board, PLAYER1, PlayerAction(2), n) == 2


def test_legal_moves():
    from agents.common import get_legal_moves, initialize_game_state

    board = initialize_game_state()
    moves = np.arange(0, 7)
    assert np.all(get_legal_moves(board) == moves)

    # board = TestBoards.board_legal_moves
    # assert get_legal_moves(board) == np.delete(moves, 2)
