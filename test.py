import timeit
from agents.common import BoardPiece, connected_four, initialize_game_state,\
    PLAYER2

c = generate_move(board, player: BoardPiece,


import timeit
number = 10**4
str_4_diag = '|=============|\n' \
                  '|X            |\n' \
                  '|X            |\n' \
                  '|O   X O X   X|\n' \
                  '|X   O O O   O|\n' \
                  '|X X O X O   X|\n' \
                  '|X O X O O O X|\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'

board_test_4 = string_to_board(str_4_diag)
board = initialize_game_state()
board[0,0] = PLAYER2
res = timeit.timeit("connected_four(board, player, last_action)",
                    setup="connected_four(board, player, last_action)",
                    number=number,
                    globals=dict(connected_four=connected_four,
                                 board=board_test_4,
                                 player=PLAYER2,
                                 last_action=0))

print(f"Python iteration-based: {res/number*1e6 : .1f} us per call")


|=============|
|             |
|             |
|             |
|             |
|             |
|X     X O    |
|=============|
|0 1 2 3 4 5 6|


|=============|
|             |
|             |
|             |
|             |
|             |
|      X O    |
|=============|
|0 1 2 3 4 5 6|


# depth = 2
|=============|
|             |
|            X|
|            X|
|            O|
|            X|
|  O O O     X|
|=============|
|0 1 2 3 4 5 6|





def heuristic_2(board: np.ndarray,
                player: BoardPiece,
                max_min: MaxMin,
                last_action) -> int:

    max_row, max_column = board.shape
    inf = np.array([10000, -10000])

    if connected_four(board, player, last_action):
        return inf[::max_min][0]

    next_player = change_player(player)

    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:
            board_new = apply_player_action(board, PlayerAction(column),
                                            next_player, True)
            if connected_four(board_new, next_player, PlayerAction(column)):
                return inf[::(-1*max_min)][0]
    else:
        last_row = 0
        # find last row
        for i in range(1, max_row+1):
            if board[max_row-i, last_action] == player:
                last_row = max_row-i
                break
        diff_center = abs(last_action - 3)
        diff_top = max_min * 5* (max_row - last_row)
        return - (diff_center-1.5) * 10 * max_min + diff_top




#next_player = change_player(player)
    #for column in range(max_column):
    #    if board[max_row-1, column] == NO_PLAYER:
    #        board_new = apply_player_action(board, PlayerAction(column),
    #                                        next_player, True)
    #        if connected_four(board_new, next_player, PlayerAction(column)):
    #            return 10000 * max_min * -1
|=============|
|             |
|             |
|      O X    |
|  O   X O O  |
|X X O X O O X|
|X X O O O X X|
|=============|
|0 1 2 3 4 5 6|
