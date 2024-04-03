import game
import tic
import keys

# debug_board = game.np.array([
#                         [0, 0, 0],
#                         [0, 2, 0],
#                         [0, 0, 0]
#                         ])
GTTT = game.Game(n=5, target=4, DEBUG_STATE=None, DEBUG_PRINT=False)
GTTT.play_game_API(agent=tic.heuristics_alpha_beta_pruning, gameid="4812")
# print(tic.minimax(curr_game=GTTT, state=GTTT.initial_state))
# GTTT.play_game(game_type="AIvAI", agents=[[tic.heuristics_alpha_beta_pruning], [tic.heuristics_alpha_beta_pruning]], first_player="X")

# debug_state = game.State(state=debug_board, score=0, turn='O', available_actions=GTTT.generate_actions(debug_board))
# print(GTTT.feature_center_control(debug_state, player='O'))
# print(tic.create_game(keys.API_KEY, keys.USER_ID, "1397", "1424", 5, 4))

################## for Testing ##################

# TODO: Make sure to delete the api-keys
x_api_key = keys.API_KEY  # Your API-KEY
user_id = keys.USER_ID  # Your ID
teamid = keys.TEAM_ID  # Your Team ID
# teamid2 = "1416"  # Enemy Team ID, 5G_UWB
teamid2 = "1415"
# gameid = "4751"   # game ID you are playing
# gameid = ""

###------- One Time Operations -------###
# create_team(x_api_key, user_id, name="5G_UWB")
# add_team_member(x_api_key, user_id, teamid, member_user_id="2638")
# remove_team_member(x_api_key, user_id, teamid, member_user_id="2638")
# get_team_member(x_api_key, user_id, teamid="1416")
# get_my_teams(x_api_key, user_id)

###------- Playing Games / Ongoing Operations -------###
# create_game(x_api_key, user_id, teamid, teamid2, board_size=5, target_num=4)                                # Success example
# create_game(x_api_key, user_id, teamid2, teamid, board_size=10, target_num=12)  # Fail example (Because target_num is bigger than the board_size)
# get_my_games(x_api_key, user_id, history_type="myGames")      # Every game you've played
# get_my_games(x_api_key, user_id, history_type="myOpenGames")  # Only Opened games
# make_move(x_api_key, user_id, gameid, teamid, where_to_move=(2,2))
# print(get_moves(x_api_key, user_id, gameid, count_most_recent_moves="1"))
# print(get_game_details(x_api_key, user_id, gameid))
# get_board_string(x_api_key, user_id, gameid)
# get_board_map(x_api_key, user_id, gameid)
# print(get_game_details(x_api_key, user_id, gameid))

# get_my_teams()
# get_board_map(x_api_key, user_id)
