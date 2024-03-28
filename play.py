import game
import tic
from typing import Callable


GTTT = game.Game(n=3, target=3)
# print(tic.minimax(curr_game=GTTT, state=GTTT.initial_state))
GTTT.play_game(
    game_type="PvAI",
    agents=[[tic.minimax]],
)
