import numpy as np
from typing import Optional, Callable
from agents.common import PlayerAction, BoardPiece, SavedState, GenMove
from agents.agent_random import generate_move

def user_move(board: np.ndarray,
              _player: BoardPiece,
              saved_state: Optional[SavedState]) -> (PlayerAction,
                                                     Optional[SavedState]):
    """
    Asks user which column should be played and returns the action

    return: PlayerAction, Optional[SavedState]
    """
    action = PlayerAction(-1)
    while not 0 <= action < board.shape[1]:
        try:
            action = PlayerAction(input("Column? "))
        except ValueError:
            print("Input could not be converted to the dtype PlayerAction, "
                  "try entering an integer.")
    return action, saved_state


def human_vs_agent(generate_move_1: GenMove,
                   generate_move_2: GenMove = user_move,
                   player_1: str = "Player 1",
                   player_2: str = "Player 2",
                   args_1: tuple = (),
                   args_2: tuple = (),
                   init_1: Callable = lambda board, player: None,
                   init_2: Callable = lambda board, player: None,) -> None:

    import time
    from agents.common import PLAYER1, PLAYER2, PLAYER1_PRINT, PLAYER2_PRINT, \
        GameState
    from agents.common import initialize_game_state, pretty_print_board, \
        apply_player_action, check_end_state

    players = (PLAYER1, PLAYER2)
    for play_first in (1, -1):
        print('play_first: ',play_first)
        for init, player in zip((init_1, init_2)[::play_first], players):
            print(list(zip((init_1, init_2)[::play_first], players)))
            print('init', init, 'player', player)
            init(initialize_game_state(), player)


human_vs_agent(user_move)
