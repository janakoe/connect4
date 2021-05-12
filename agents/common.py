from enum import Enum
import numpy as np
from typing import Optional, Callable, Tuple


# board[i, j] == PLAYER2 where max_player 2 (max_player to move second) has a piece,
# board[i, j] == PLAYER1 where max_player 1 (max_player to move first) has a piece
# board[i, j] == NO_PLAYER where the position is empty

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)
PLAYER1 = BoardPiece(1)
PLAYER2 = BoardPiece(2)

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


class SavedState:
    pass


# defines type of function (GenMove) with inputs and outputs
GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece,
    initialized to 0 (NO_PLAYER).

    return: ndarray
    """
    board = np.empty((6, 7), dtype=BoardPiece)
    board.fill(NO_PLAYER)
    return board


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout).
    The piece in board[0, 0] should appear in the lower-left. Here's an
    example output, note that we use PLAYER1_Print to represent PLAYER1 and
    PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    board_print = np.empty_like(board, dtype=BoardPiecePrint)
    board_print[board == NO_PLAYER] = NO_PLAYER_PRINT
    board_print[board == PLAYER1] = PLAYER1_PRINT
    board_print[board == PLAYER2] = PLAYER2_PRINT
    board_print = np.flip(board_print, 0)

    top = '|=============|\n'
    bottom = '|=============|\n|0 1 2 3 4 5 6|'

    pp_board = top
    for row in board_print:
        pp_board += '|'+' '.join(row)+'|\n'
    pp_board += bottom

    return pp_board


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the
    last board state as a string.
    """
    top = '|=============|\n'
    bottom = '|=============|\n|0 1 2 3 4 5 6|'

    pp_board = pp_board[len(top):(len(pp_board)-len(bottom))-1]
    pp_board = np.array(pp_board.split('\n'))
    print(pp_board)

    board_print = np.empty((6, 7), dtype=BoardPiecePrint)

    for i in range(len(board_print)):
        row = pp_board[i][1:-1:2]
        for j in range(len(board_print[0])):
            board_print[i, j] = row[j]

    board = np.empty_like(board_print, dtype=BoardPiece)
    board[board_print == NO_PLAYER_PRINT] = NO_PLAYER
    board[board_print == PLAYER1_PRINT] = PLAYER1
    board[board_print == PLAYER2_PRINT] = PLAYER2
    board = np.flip(board, 0)

    return board


def apply_player_action(board: np.ndarray,
                        action: PlayerAction,
                        player: BoardPiece,
                        copy: bool = False) -> np.ndarray:
    """
    Sets board[i, action] = max_player, where i is the lowest open row.
    The modified board is returned.
    If copy is True, makes a copy of the board before modifying it.
    """

    if copy:
        board = board.copy()

    for row in board:
        if row[action] == NO_PLAYER:
            row[action] = player
            return board

    print(f'action {action} not possible')
    return board


def connected_four(board: np.ndarray,
                   player: BoardPiece,
                   last_action: PlayerAction,
                   n: int = 4) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `max_player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False
    otherwise. If desired, the last action taken (i.e. last column played)
    can be provided for potential speed optimisation.
    """

    max_row = board.shape[0]
    max_column = board.shape[1]

    last_row = 0

    # find last row
    for i in range(1, max_row+1):
        if board[max_row-i, last_action] == player:
            last_row = max_row-i
            break

    max_connect = 0

    # check vertical
    if last_row-3 >= 0:
        for i in range(1, last_row+1):
            if board[last_row-i, last_action] != player: break
            else:
                max_connect += 1
                i += 1
                if max_connect == 3: return True

    # check horizontal
    max_connect = 0
    for i in range(1, last_action+1):
        if board[last_row, last_action-i] != player: break
        else:
            max_connect += 1
            i += 1
            if max_connect == 3: return True

    for j in range(1, max_column-last_action):
        if board[last_row, last_action+j] != player: break
        else:
            max_connect += 1
            j += 1
            if max_connect == 3: return True

    # check diagonal same direction
    max_connect = 0
    for i in range(1, min(last_row, last_action)+1):
        if board[last_row-i, last_action-i] != player: break
        else:
            i += 1
            max_connect += 1
            if max_connect == 3: return True

    for j in range(1, min(max_row-last_row, max_column-last_action)):
        if board[last_row+j, last_action+j] != player: break
        else:
            max_connect += 1
            j += 1
            if max_connect == 3: return True

    # check diagonal different direction
    max_connect = 0
    for i in range(1, min(last_row+1, max_column-last_action)):
        if board[last_row-i, last_action+i] != player: break
        else:
            max_connect += 1
            i += 1
            if max_connect == 3: return True

    for j in range(1, min(max_row-last_row, last_action+1)):
        if board[last_row+j, last_action-j] != player: break
        else:
            max_connect += 1
            j += 1
            if max_connect == 3: return True

    return False


def connected_n(board: np.ndarray,
                player: BoardPiece,
                last_action: PlayerAction,
                n: int = 4) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `max_player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False
    otherwise. If desired, the last action taken (i.e. last column played)
    can be provided for potential speed optimisation.
    """
    count = 0
    max_row = board.shape[0]
    max_column = board.shape[1]
    n_connected = n
    max_connected = 4
    n_empty = max_connected - n_connected
    last_row = 0

    # find last row
    for i in range(1, max_row+1):
        if board[max_row-i, last_action] == player:
            last_row = max_row-i
            break

    connected = 1
    # check vertical
    if last_row - n_connected >= 0:
        for i in range(1, last_row+1):
            if board[last_row-i, last_action] != player:
                break
            if i <= (max_connected - n_connected):
                if last_row+i >= max_row:
                    break
            else:
                connected += 1
                i += 1
                if connected == n_connected:
                    count += 1

    # check horizontal
    connected = 1
    empty = 0

    for i in range(1, min(last_action+1, max_connected)):
        if board[last_row, last_action-i] == player:
            connected += 1
            i += 1
        elif board[last_row, last_action-i] == NO_PLAYER:
            if empty >= n_empty:
                break
            empty += 1
            i += 1
        else:
            break
    if (connected == n_connected) & (empty == n_empty):
        count += 1

    for j in range(1, max_column-last_action):
        if board[last_row, last_action+j] == player:
            connected += 1
            if (connected == n_connected) & (empty == n_empty):
                count += 1
            j += 1
        elif board[last_row, last_action+j] == NO_PLAYER:
            if empty >= n_empty:
                break
            empty += 1
            if (connected == n_connected) & (empty == n_empty):
                count += 1
            j += 1
        else:
            break

    # check diagonal same direction
    connected = 1
    empty = 0
    for i in range(1, min(max_connected, min(last_row, last_action)+1)):
        if board[last_row-i, last_action-i] == player:
            i += 1
            connected += 1
        elif board[last_row-i, last_action-i] == NO_PLAYER:
            if empty >= n_empty:
                break
            i += 1
            empty += 1
        else:
            break

    if (connected == n_connected) & (empty == n_empty):
        count += 1

    for j in range(1, min(max_row-last_row, max_column-last_action)):
        if board[last_row+j, last_action+j] == player:
            connected += 1
            if connected == n_connected & empty == n_empty:
                count += 1
            j += 1
        elif board[last_row+j, last_action+j] == NO_PLAYER:
            if empty >= n_empty:
                break
            empty += 1
            j += 1
            if (connected == n_connected) & (empty == n_empty):
                count += 1
        else:
            break

    # check diagonal different direction
    connected = 1
    empty = 0
    for i in range(1, min(last_row+1, max_column-last_action)):
        if board[last_row-i, last_action+i] == player:
            i += 1
            connected += 1
        elif board[last_row-i, last_action+i] == NO_PLAYER:
            if empty >= n_empty:
                break
            i += 1
            empty += 1
        else:
            break

    if (connected == n_connected) & (empty == n_empty):
        count += 1

    for j in range(1, min(max_row-last_row, last_action+1)):
        if board[last_row+j, last_action-j] == player:
            connected += 1
            if (connected == n_connected) & (empty == n_empty):
                count += 1
            j += 1
        elif board[last_row+j, last_action-j] == NO_PLAYER:
            if empty >= n_empty:
                break
            empty += 1
            j += 1
            if (connected == n_connected) & (empty == n_empty):
                count += 1
        else:
            break

    return count

def check_end_state(board: np.ndarray,
                    player: BoardPiece,
                    last_action: PlayerAction,) -> GameState:
    """
    Returns the current game state for the current `max_player`, i.e. has their
    last action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player, last_action):
        return GameState.IS_WIN
    elif np.all(board != NO_PLAYER):
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING


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
