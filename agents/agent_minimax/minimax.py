import numpy as np
from agents.common import BoardPiece, SavedState, PlayerAction, \
    apply_player_action, connected_four, NO_PLAYER, PLAYER1, PLAYER2, connected_n
from typing import Tuple, Optional


MaxMin = np.int8
MAX = MaxMin(1)
MIN = MaxMin(-1)


def generate_move(board: np.ndarray,
                  player: BoardPiece,
                  saved_state: Optional[SavedState]) -> Tuple[PlayerAction,
                                                              Optional[SavedState]]:

    depth = 5
    _, column = max_player_move(board, player, depth-1)
    return PlayerAction(column), saved_state


def max_player_move(board: np.ndarray,
                    max_player: BoardPiece,
                    depth: int) -> (int, int):

    max_row, max_column = board.shape
    scores = np.zeros(max_column)

    # termination condition recursion: evaluate heuristic of board
    if depth == 0:
        for column in range(max_column):
            # if action valid apply max_player action, else score = None
            if board[max_row-1, column] != NO_PLAYER:
                scores[column] = np.nan
            else:
                board_new = apply_player_action(board, PlayerAction(column), max_player, True)
                if connected_four(board_new, max_player, PlayerAction(column)):
                    return np.inf, column
                scores[column] = heuristic(board_new, max_player, MAX, PlayerAction(column))
        return np.nanmax(scores), np.nanargmax(scores)

    # iterate over all columns and calculate future moves by calling
    # minimizing max_player with depth-1
    for column in range(max_column):
        # if action valid apply max_player action, else score = None
        if board[max_row-1, column] != NO_PLAYER:
            scores[column] = np.nan
        else:
            board_new = apply_player_action(board, PlayerAction(column),
                                            max_player, copy=True)
            # check if max_player wins by this column
            if connected_four(board_new, max_player, PlayerAction(column)):
                # scores[column] = np.inf
                return np.inf, column
            else:
                min_player = change_player(max_player)
                scores[column], idx = min_player_move(board_new, min_player,
                                                      depth-1)
    return np.nanmax(scores), np.nanargmax(scores)


def min_player_move(board: np.ndarray, min_player: BoardPiece, depth: int) -> \
        (int, int):

    max_row, max_column = board.shape
    scores = np.zeros(max_column)

    # termination condition recursion: evaluate heuristic of board
    if depth == 0:
        for column in range(max_column):
            # if action valid apply max_player action, else score = None
            if board[max_row-1, column] != NO_PLAYER:
                scores[column] = np.nan
            else:
                board_new = apply_player_action(board, PlayerAction(column), min_player, True)
                # TO DO check wins first
                if connected_four(board_new, min_player, PlayerAction(column)):
                    return np.NINF, column
                scores[column] = heuristic(board_new, min_player, MIN, PlayerAction(column))
        return np.nanmin(scores), np.nanargmin(scores)

    # iterate over all columns and calculate future moves by calling
    # maximizing max_player with depth+1
    for column in range(max_column):
        # if action valid apply max_player action, else score = None
        if board[max_row-1, column] != NO_PLAYER:
            scores[column] = np.nan
        else:
            # check if max_player wins by this column
            board_new = apply_player_action(board, PlayerAction(column), min_player, True)
            if connected_four(board_new, min_player, PlayerAction(column)):
                # scores[column] = np.NINF
                return np.NINF, column
            else:
                max_player = change_player(min_player)
                scores[column], idx = max_player_move(board_new, max_player,
                                                      depth-1)

    return np.nanmin(scores), np.nanargmin(scores)


def heuristic_2(board: np.ndarray,
              player: BoardPiece,
              max_min: MaxMin,
              last_action) -> int:

    #MaxMin = 1 for max, -1 for min
    # which last action!! Make sure its the initial column
    max_row, max_column = board.shape
    inf = np.array([np.inf, np.NINF])

    if connected_four(board, player, last_action):
        return inf[::max_min][0]

    next_player = change_player(player)

    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:
            board_new = apply_player_action(board, PlayerAction(column), next_player, True)
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


def change_player(player):
    if player == PLAYER1:
        return PLAYER2
    return PLAYER1


def heuristic(board: np.ndarray,
              player: BoardPiece,
              max_min: MaxMin,
              last_action) -> int:

    #MaxMin = 1 for max, -1 for min
    # which last action!! Make sure its the initial column
    max_row, max_column = board.shape
    inf = np.array([np.inf, np.NINF])

    if connected_four(board, player, last_action):
        return inf[::max_min][0]

    next_player = change_player(player)

    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:
            board_new = apply_player_action(board, PlayerAction(column), next_player, True)
            if connected_four(board_new, next_player, PlayerAction(column)):
                return inf[::(-1*max_min)][0]
    else:
        number_3 = connected_n(board, player, last_action, n=3)
        number_2 = connected_n(board, player, last_action, n=2)
        return (number_3 * 100 + number_2 * 10) * max_min


