GTTT = game.Game(n=3, target=3, DEBUG_STATE=debug_board, DEBUG_PRINT=False)
# GTTT.play_game_API(agent=tic.minimax)
# print(tic.minimax(curr_game=GTTT, state=GTTT.initial_state))
GTTT.play_game(game_type="PvAI", agents=[[tic.minimax]], first_player="O")
