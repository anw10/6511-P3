import game
import tic
from typing import Callable

debug_board = (
                game.np.array([
                    [0, 1, 0],
                    [2, 1, 0],
                    [0, 0, 0]
                ])
                ,
                'O'
            )
GTTT = game.Game(n=3, target=3, DEBUG_STATE=None, DEBUG_PRINT=False)
# print(tic.minimax(curr_game=GTTT, state=GTTT.initial_state))
GTTT.play_game(
    game_type="PvAI",
    agents=[[tic.minimax]],
    first_player='X'
)
