import game
import tic

debug_board = game.np.array([[2, 2, 0], [0, 1, 0], [1, 0, 1]])
GTTT = game.Game(n=3, target=3, DEBUG_STATE=None, DEBUG_PRINT=False)
# print(tic.minimax(curr_game=GTTT, state=GTTT.initial_state))
GTTT.play_game(game_type="PvAI", agents=[[tic.minimax]], first_player="X")
