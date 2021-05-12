import numpy as np
from ..common import BoardPiece, SavedState, PlayerAction, NO_PLAYER
from typing import Tuple, Optional


def generate_move_random(board: np.ndarray,
                         player: BoardPiece,
                         saved_state: Optional[SavedState]) -> \
        Tuple[PlayerAction, Optional[SavedState]]:

    max_row = board.shape[0]
    column = np.random.randint(0, np.shape(board)[1])

    while board[max_row-1, column] != NO_PLAYER:
        column = np.random.randint(0, np.shape(board)[1])

    return PlayerAction(column), saved_state
