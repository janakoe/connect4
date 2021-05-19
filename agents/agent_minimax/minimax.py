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
    """
    Generates move for player by calling the recursive
    function max_player_move with depth 7, alpha=-99999 and beta=99999

    Parameters
    ----------
    board
        current game board
    player
        player for whom a move is generated
    saved_state
        not used
    Returns
    -------
    PlayerAction
        column to be played
    saved_state
        not used
    """

    depth = 7
    score, column = max_player_move(board, player, depth, -99999, 99999)
    return PlayerAction(column), saved_state


def max_player_move(board: np.ndarray,
                    max_player: BoardPiece,
                    depth: int,
                    alpha: int,
                    beta: int) -> Tuple[int, int]:
    """
    Finds the best move for the maximizing player: the column to
    be played in order to reach a maximal score. If depth equals zero,
    the board is evaluated by calling evaluate_max. If depth is unequal zero,
    the function loops over all possible moves, makes the move and then
    calls the function to generates the next move for the minimizing player
    with depth-1.
    If a win happens the score 10000 * depth is returned, such that early wins
    are more favorable.

    Parameters
    ----------
    board: np.ndarray
        current game board (copy)
    max_player: BoardPiece
        current player (player one or two)
    depth: int
        tree depth
    alpha: int
        current minimal score for maximizing player
    beta: int
        current maximal score for minimizing player

    Returns
    -------
    max_score: int
        maximal score that maximizing player can reach
    player_action: int
        column to be played - PlayerAction(column) that yields maximal score
    """
    max_row, max_column = board.shape
    max_score = -99999
    player_action = -1
    possible_action = np.argwhere(board[max_row-1, :] == NO_PLAYER).flatten()
    try:
        player_action = np.random.choice(possible_action)
    except ValueError:
        max_score = -10000

    # termination condition recursion: evaluate heuristic of board
    if depth == 0:
        return evaluate_max(board, max_player, alpha, beta)

    # check for immediate wins to speed up
    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:
            board_new = apply_player_action(board, PlayerAction(column),
                                            max_player, copy=True)
            if connected_four(board_new, max_player, PlayerAction(column)):
                return 10000 * depth, PlayerAction(column)

    # loop over columns and apply player action if column not full
    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:
            board_new = apply_player_action(board, PlayerAction(column),
                                            max_player, copy=True)

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

    return max_score, PlayerAction(player_action)


def min_player_move(board: np.ndarray, min_player: BoardPiece, depth: int,
                    alpha: int, beta: int) -> Tuple[int, int]:
    """
    Finds the best move for the minimizing player: the column to
    be played in order to reach a minimal score. If depth equals zero,
    the board is evaluated by calling evaluate_min. If depth is unequal zero,
    the function loops over all possible moves, makes the move and then
    calls the function to generates the next move for the maximizing player
    with depth-1.
    If a win happens the score -10000 * depth is returned, such that early
    wins are more favorable.

    Parameters
    ----------
    board: np.ndarray
        current game board (copy)
    min_player: BoardPiece
        current player (player one or two)
    depth: int
        tree depth
    alpha: int
        current minimal score for maximizing player
    beta: int
        current maximal score for minimizing player

    Returns
    -------
    max_score: int
        maximal score that maximizing player can reach
    player_action: int
        column to be played - PlayerAction(column) that yields maximal score
    """

    max_row, max_column = board.shape
    min_score = 99999
    player_action = -1
    possible_action = np.argwhere(board[max_row-1, :] == NO_PLAYER).flatten()
    try:
        player_action = np.random.choice(possible_action)
    except ValueError:
        min_score = 10000

    # termination condition recursion: evaluate heuristic of board
    if depth == 0:
        return evaluate_min(board, min_player, alpha, beta)

    # check for immediate wins - for speed up
    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:
            board_new = apply_player_action(board, PlayerAction(column),
                                            min_player, copy=True)
            if connected_four(board_new, min_player, PlayerAction(column)):
                return -10000 * depth, PlayerAction(column)

    # loop over columns and apply player action if column not full
    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:
            board_new = apply_player_action(board, PlayerAction(column),
                                            min_player, copy=True)

            # call max_player_move to get score of column
            max_player = change_player(min_player)
            score, _ = max_player_move(board_new, max_player, depth-1,
                                       alpha, beta)

            # update max_score and player_action
            if score < min_score:
                min_score = score
                player_action = column
                if min_score < beta:
                    beta = min_score
                if beta <= alpha:
                    break

    return min_score, PlayerAction(player_action)


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


def heuristic(board: np.ndarray,
              player: BoardPiece,
              max_min: MaxMin,
              last_action) -> int:
    """
    Evaluates the board for the given player in respect to the
    last column played (last_action). The achieved score for the player by
    playing that column is returned. If the player wins by the last action,
    9999 * max_min is returned. Otherwise it is calculated if there are three
    and or two connected board pieces and if so how many. The score is then
    calculated as follows:
    (number_connected_3 * 100 + number_connected_2 * 10) * max_min

    Parameters
    ----------
    board: np.ndarray
        current game board (copy)
    player: BoardPiece
        current player (player one or two)
        in respect to which the board should be evaluated
    max_min: MaxMin
        whether player is the minimizing (max_min = -1)
        or maximizing player (max_min = 1)
    last_action: PlayerAction
        last column played by player

    Returns
    -------
    int score
    """

    if connected_four(board, player, last_action):
        return (10000-1) * max_min

    number_connected_3 = connected_n(board, player, last_action, n=3)
    number_connected_2 = connected_n(board, player, last_action, n=2)

    return (number_connected_3 * 100 + number_connected_2 * 10) * max_min


def evaluate_max(board: np.ndarray,
                 max_player: BoardPiece,
                 alpha: int,
                 beta: int) -> Tuple[int, int]:
    """
    Loops over all columns and calculates the score that would be achieved
    by the maximizing player when playing this column by calling the
    heuristic function.
    The maximal score and the corresponding action (column) is returned.

    Parameters
    ----------
    board: np.ndarray
        current game board (copy)
    max_player: BoardPiece
        current player (player one or two)
    alpha: int
        current minimal score for maximizing player
    beta: int
        current maximal score for minimizing player

    Returns
    -------
    max_score: int
        maximal score that maximizing player can reach
    player_action: int
        column to be played - PlayerAction(column) that yields maximal score
    """
    max_row, max_column = board.shape
    max_score = -99999
    max_action = -1
    possible_action = np.argwhere(board[max_row-1, :] == NO_PLAYER).flatten()
    try:
        max_action = np.random.choice(possible_action)
    except ValueError:
        max_score = -10000

    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:
            # player action
            board_new = apply_player_action(board, PlayerAction(column),
                                            max_player, True)
            # check for win
            if connected_four(board_new, max_player, PlayerAction(column)):
                return 10000, PlayerAction(column)
            # evaluate score
            score = heuristic(board_new, max_player, MAX, PlayerAction(column))

            if score > max_score:
                max_score = score
                max_action = column
                if max_score > alpha:
                    alpha = max_score
                if alpha >= beta:
                    break

    return max_score, PlayerAction(max_action)


def evaluate_min(board: np.ndarray,
                 min_player: BoardPiece,
                 alpha: int,
                 beta: int) -> Tuple[int, int]:
    """
    Loops over all columns and calculates the score that would be achieved
    by the minimizing player when playing this column by calling the
    heuristic function.
    The minimal score and the corresponding action (column) is returned.

    Parameters
    ----------
    board: np.ndarray
        current game board (copy)
    min_player: BoardPiece
        current player (player one or two)
    alpha: int
        current minimal score for maximizing player
    beta: int
        current maximal score for minimizing player

    Returns
    -------
    min_score: int
        minimal score that maximizing player can reach
    player_action: int
        column to be played - PlayerAction(column) that yields minimal score
    """

    max_row, max_column = board.shape
    min_score = 99999
    min_action = -1
    possible_action = np.argwhere(board[max_row-1, :] == NO_PLAYER).flatten()
    try:
        min_action = np.random.choice(possible_action)
    except ValueError:
        min_score = 10000

    for column in range(max_column):
        if board[max_row-1, column] == NO_PLAYER:

            board_new = apply_player_action(board, PlayerAction(column),
                                            min_player, True)
            if connected_four(board_new, min_player, PlayerAction(column)):
                return -10000, PlayerAction(column)

            score = heuristic(board_new, min_player, MIN, PlayerAction(column))

            if score < min_score:
                min_score = score
                min_action = column
                if min_score < beta:
                    beta = min_score
                if beta <= alpha:
                    break
    return min_score, PlayerAction(min_action)
