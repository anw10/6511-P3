import game
import tic
from typing import Callable

debug_board = game.np.array([
    [1, 0, 2],
    [2, 1, 1],
    [0, 1, 2]
])
GTTT = game.Game(n=3, target=3, DEBUG_STATE=debug_board, DEBUG_PRINT=True)
# print(tic.minimax(curr_game=GTTT, state=GTTT.initial_state))
GTTT.play_game(
    game_type="PvAI",
    agents=[[tic.minimax]],
    first_player='O'
)
