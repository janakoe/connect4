from enum import Enum
import numpy as np
from typing import Optional, Callable, Tuple
from numba import njit

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
    Initializes a new empty game board as a ndarray with shape (6,7) and
    data type BoardPiece initialized to 0 (NO_PLAYER).

    return: ndarray
        empty board, shape: 6x7
    """
    board = np.empty((6, 7), dtype=BoardPiece)
    board.fill(NO_PLAYER)
    return board


def pretty_print_board(board: np.ndarray) -> str:
    """
    Return the game board converted to a human readable string representation,
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

    Parameters
    ----------
    board: np.ndarray
        game board in array representation

    Returns
    -------
    str
        game board in string representation

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

    Parameters
    ----------
    pp_board
    """
    top = '|=============|\n'
    bottom = '|=============|\n|0 1 2 3 4 5 6|'

    pp_board = pp_board[len(top):(len(pp_board)-len(bottom))-1]
    pp_board = np.array(pp_board.split('\n'))

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
    If action is not valid, NameError is raised.
    Parameters
    ----------
    board: np.ndarray
        array representation of the current board
    action: PlayerAction
        column to be played by player
    player: BoardPiece
        player who makes the move
    copy: bool

    Returns
    -------
    new board: np.ndarray 

    """

    if copy:
        board = board.copy()

    for row in board:
        if row[action] == NO_PLAYER:
            row[action] = player
            return board

    print(pretty_print_board(board))
    raise NameError(f'action {action} not possible')


@njit()
def connected_four(board: np.ndarray,
                   player: BoardPiece,
                   last_action: PlayerAction) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `max_player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False
    otherwise. If desired, the last action taken (i.e. last column played)
    can be provided for potential speed optimisation.

    Parameters
    ----------
    board: np.ndarray
        array representation of the current board
    player: BoardPiece
        player who did the last move
    last_action: PlayerAction
        last action of the player (last column that was played)

    Returns
    -------
    bool
        four connected pieces True or False

    """

    max_row = board.shape[0]
    max_column = board.shape[1]

    last_row = 0

    if last_action is None:
        return False

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


@njit()
def connected_n(board: np.ndarray,
                player: BoardPiece,
                last_action: PlayerAction,
                n: int) -> int:
    """
    Returns number of cases in which n pieces equal to `max_player`
    arranged in either a horizontal, vertical, or diagonal line only if it is
    still possible for the player to connect four pieces. Therefore the
    function checks for 4-n remaining free places (n_empty) with no board
    boundaries.

    Parameters
    ----------
    board: np.ndarray
        array representation of the current board
    player: BoardPiece
        player who did the last move
    last_action: PlayerAction
        last action of the player (last column that was played)
    n: int
        number of connected pieces

    Returns
    -------
    count: int
        number of n_connected sequences
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

    # check vertical
    connected = 1
    if (last_row - n_connected >= -1) & (last_row + n_empty < max_row):
        for i in range(1, last_row+1):
            if board[last_row-i, last_action] != player:
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

    # check diagonal same directions
    connected = 1
    empty = 0
    for i in range(1, min(last_row, last_action)+1):
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
            if (connected == n_connected) & (empty == n_empty):
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

    # check diagonal different directions
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
                    last_action: PlayerAction) -> GameState:
    """
    Returns the current game state for the current `max_player`, i.e. did their
    last action lead to a win (GameState.IS_WIN) or drawn (GameState.IS_DRAW)
    or is the game still on-going (GameState.STILL_PLAYING)

    Parameters
    ----------
    board: np.ndarray
        game board in array representation to check for state
    player: BoardPiece
        player who did the last move
    last_action: PlayerAction
        last action of the player (last column that was played)

    Returns
    -------
    GameState
    """
    if connected_four(board, player, last_action):
        return GameState.IS_WIN
    elif np.all(board != NO_PLAYER):
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING


def change_player(player: BoardPiece) -> BoardPiece:
    """
    Changes the player and returns the new player
    Parameters
    ----------
    player: BoardPiece
        current player (player one or two)

    Returns
    -------
    BoardPiece
        new player
    """

    if player == PLAYER1:
        return PLAYER2
    return PLAYER1


def valid_action(board: np.ndarray) -> np.ndarray:
    """
    Returns valid actions of the given board, e.g. all columns that are not
    full.

    Parameters
    ----------
    board: np.ndarray
        current game board
    Returns
    -------
    np.array of valid columns
    """
    return np.argwhere(board[-1, :] == NO_PLAYER).flatten()
