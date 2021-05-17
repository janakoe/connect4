import numpy as np
from agents.common import BoardPiece, SavedState, PlayerAction, \
    apply_player_action, connected_four, NO_PLAYER, PLAYER1, PLAYER2, connected_n
from typing import Tuple, Optional

MaxMin = np.int8
MAX = MaxMin(1)
MIN = MaxMin(-1)


def generate_move(board: np.ndarray,
                  player: BoardPiece,
                  saved_state: Optional[SavedState])\
                    -> Tuple[PlayerAction, Optional[SavedState]]:

    depth = 8
    score, column = max_player_move(board, player, depth-1, -99999, 99999)
    return PlayerAction(column), saved_state


def max_player_move(board: np.ndarray,
                    max_player: BoardPiece,
                    depth: int,
                    alpha: int,
                    beta: int):

    max_row, max_column = board.shape
    possible_action = np.argwhere(board[max_row-1,:] == NO_PLAYER).flatten()
    player_action = np.random.choice(possible_action)
    max_score = -99999

    # termination condition recursion: evaluate heuristic of board
    if depth == 0:
        return evaluate_max(board, max_player, alpha, beta)

    # loop over columns and apply player action if column not full
    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:
            board_new = apply_player_action(board, PlayerAction(column),
                                            max_player, copy=True)
            # check if max_player wins by this column
            if connected_four(board_new, max_player, PlayerAction(column)):
                return 10000 * depth, column
            # call min_player_move to get score of column
            min_player = change_player(max_player)
            score, _ = min_player_move(board_new, min_player, depth-1,
                                       alpha, beta)

            # update max_score and player_action
            if score > max_score:
                max_score = score
                player_action = column
                if max_score > alpha:
                    alpha = max_score
                if alpha >= beta:
                    break
    return max_score, player_action


def min_player_move(board: np.ndarray, min_player: BoardPiece, depth: int,
                    alpha: int, beta: int) -> (int, int):

    max_row, max_column = board.shape
    possible_action = np.argwhere(board[max_row-1,:] == NO_PLAYER).flatten()
    player_action = np.random.choice(possible_action)
    min_score = 99999

    # termination condition recursion: evaluate heuristic of board
    if depth == 0:
        return evaluate_min(board, min_player, alpha, beta)

    # loop over columns and apply player action if column not full
    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:
            board_new = apply_player_action(board, PlayerAction(column),
                                            min_player, copy=True)
            # check if max_player wins by this column
            if connected_four(board_new, min_player, PlayerAction(column)):
                return -10000 * depth, column
            # call min_player_move to get score of column
            max_player = change_player(min_player)
            score,_ = max_player_move(board_new, max_player, depth-1,
                                      alpha, beta)
            # update max_score and player_action

            if score < min_score:
                min_score = score
                player_action = column
                if min_score < beta:
                    beta = min_score
                if beta <= alpha:
                    break
    return min_score, player_action


def change_player(player):
    if player == PLAYER1:
        return PLAYER2
    return PLAYER1


def heuristic(board: np.ndarray,
              player: BoardPiece,
              max_min: MaxMin,
              last_action) -> int:

    if connected_four(board, player, last_action):
        return (10000-1) * max_min

    number_3 = connected_n(board, player, last_action, n=3)
    number_2 = connected_n(board, player, last_action, n=2)

    return (number_3 * 100 + number_2 * 10) * max_min


def evaluate_max(board: np.ndarray,
                 max_player: BoardPiece,
                 alpha: int,
                 beta: int) -> [int, int]:

    max_row, max_column = board.shape
    max_score = -99999
    possible_action = np.argwhere(board[max_row-1,:] == NO_PLAYER).flatten()
    max_action = np.random.choice(possible_action)

    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:
            # player action
            board_new = apply_player_action(board, PlayerAction(column),
                                            max_player, True)
            # check for win
            if connected_four(board_new, max_player, PlayerAction(column)):
                return 10000, column
            # evaluate score
            score = heuristic(board_new, max_player, MAX, PlayerAction(column))

            if score > max_score:
                max_score = score
                max_action = column
                if max_score > alpha:
                    alpha = max_score
                if alpha >= beta:
                    break

    return max_score, max_action


def evaluate_min(board: np.ndarray,
                 min_player: BoardPiece,
                 alpha: int,
                 beta: int) -> [int, int]:

    max_row, max_column = board.shape
    min_score = 99999
    possible_action = np.argwhere(board[max_row-1,:] == NO_PLAYER).flatten()
    min_action = np.random.choice(possible_action)

    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:

            board_new = apply_player_action(board, PlayerAction(column),
                                            min_player, True)
            if connected_four(board_new, min_player, PlayerAction(column)):
                return -10000, column

            score = heuristic(board_new, min_player, MIN, PlayerAction(column))

            if score < min_score:
                min_score = score
                min_action = column
                if min_score < beta:
                    beta = min_score
                if beta <= alpha:
                    break
    return min_score, min_action
