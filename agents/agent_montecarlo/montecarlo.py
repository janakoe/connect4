import numpy as np
from agents.common import BoardPiece, SavedState, PlayerAction, \
    apply_player_action, connected_four, NO_PLAYER, PLAYER1, PLAYER2, \
    connected_n
from typing import Tuple, Optional

 # Get the best move from available statistics.
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
    raise NotImplementedError
    # return PlayerAction(column), saved_state

# From given state, repeatedly run MCTS to build statistics.
def runSearch(state, timeout):
    raise NotImplementedError





class MonteCarloNode:

    def __init__(self, parent, play, state, unexpandedPlays):
        #pass
        self.play = play
        self.state = state

        # Monte Carlo stuff
        self.n_plays = 0
        self.n_wins = 0

        # Tree stuff
        self.parent = parent
        #self.children = new Map()

        for legal_play in unexpandedPlays:
            self.children.set(play.hash(), { play: play, node: None })




