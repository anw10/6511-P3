import game
import tic
import keys

# debug_board = game.np.array([
#                         [1, 2, 0],
#                         [1, 0, 0],
#                         [0, 0, 0]
#                         ])
GTTT = game.Game(n=5, target=4, DEBUG_STATE=None, DEBUG_PRINT=False)
# GTTT.play_game_API(agent=tic.minimax)
# print(tic.minimax(curr_game=GTTT, state=GTTT.initial_state))
GTTT.play_game(game_type="PvAI", agents=[[tic.minimax]], first_player="O")

# print(tic.create_game(keys.API_KEY, keys.USER_ID, "1397", "1424", 5, 4))
