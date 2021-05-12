import numpy as np
# import pytest
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, \
    PLAYER2, PlayerAction


def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
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
    from agents.common import apply_player_action

    raise NotImplementedError
    # apply_player_action(board, action, max_player)

    # test if origin board unchanged
    # new_board = apply_player_action(board, action, max_player, True)
    # assert new_board != board



def test_string_to_board():
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


def test_boards(i):
    from agents.common import string_to_board
    str_board_weird_win = '|=============|\n' \
                  '|X            |\n' \
                  '|X            |\n' \
                  '|O   X O X   X|\n' \
                  '|X   O O O   O|\n' \
                  '|X X O X O   X|\n' \
                  '|X O X O O O X|\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'
    board_weird_win = string_to_board(str_board_weird_win)
    return board_weird_win


def test_connected_four():
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

    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    i = np.random.randint(3, 6)
    j = np.random.randint(0, 4)
    board[np.arange(i, i-3, -1), np.arange(j, j+3)] = player
    print(board)
    assert not connected_four(board, player, PlayerAction(j))

    board = test_boards()
    player = PLAYER1
    i = 1
    assert not connected_four(board, player, PlayerAction(i))


def test_connected_n():
    from agents.common import connected_n, initialize_game_state
    n = 3
    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])
    board[1,1]= player

    #board[np.arange(i, i-(n-1), -1), np.arange(j, j+(n-1))] = player
    print(board)
    test = connected_n(board, player, PlayerAction(0), n)
    print(test)
    #assert not connected_n(board, player, PlayerAction(1), n)



