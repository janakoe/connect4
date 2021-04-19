import numpy as np
from ..common import BoardPiece, SavedState, PlayerAction
from typing import Callable, Tuple, Optional


def generate_move_random(board: np.ndarray,
                         player: BoardPiece,
                         saved_state: Optional[SavedState]) -> \
        Tuple[PlayerAction, Optional[SavedState]]:

    column = np.random.randint(0, np.shape(board)[1])

    return PlayerAction(column), saved_state
