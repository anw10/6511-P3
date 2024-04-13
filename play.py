import game
import keys
from tic import *

################## For Playing ##################

###------- Keys -------###

x_api_key = keys.API_KEY    # Your API-KEY
user_id = keys.USER_ID      # Your ID
teamid = keys.TEAM_ID       # Your Team ID
# teamid_5G = "1416"
# teamid2 = "1404"              # Opponent Team ID
# gameid = "5173"               # Current game

###------- Start Game -------###

# debug_board = np.array([
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 2, 0, 0],
#     [0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0]
# ])

# debug_state = game.State(state=debug_board, score=0, turn='O', available_actions=[])

# GTTT = game.Game(n=5, target=4, DEBUG_STATE=None, DEBUG_PRINT=False)                              # IMPORTANT: Initialize n and target m correctly
# GTTT.play_game(game_type="PvAI", agents=[[heuristics_alpha_beta_pruning]], first_player="O")      # Local play
# GTTT.play_game_API(agent=heuristics_alpha_beta_pruning, gameid=gameid)                            # Play via API

################## For Running One Time API Calls ##################

###------- One Time Operations -------###
# create_team(x_api_key, user_id, name="5G_UWB")
# add_team_member(x_api_key, user_id, teamid, member_user_id="2638")
# remove_team_member(x_api_key, user_id, teamid, member_user_id="2638")
# get_team_member(x_api_key, user_id, teamid="1416")
# get_my_teams(x_api_key, user_id)

###------- Playing Games / Ongoing Operations -------###
# create_game(x_api_key, user_id, teamid, teamid_5G, board_size=5, target_num=4)                                # Success example
# create_game(x_api_key, user_id, teamid2, teamid, board_size=10, target_num=12)  # Fail example (Because target_num is bigger than the board_size)
# get_my_games(x_api_key, user_id, history_type="myGames")      # Every game you've played
# get_my_games(x_api_key, user_id, history_type="myOpenGames")  # Only Opened games
# make_move(x_api_key, user_id, gameid, teamid_5G, where_to_move=(4,0))

# print(get_game_details(x_api_key, user_id, gameid))
# get_board_string(x_api_key, user_id, gameid)
# get_board_map(x_api_key, user_id, gameid)
# print(get_game_details(x_api_key, user_id, gameid))
# get_my_teams()
# get_board_map(x_api_key, user_id)
# print(get_moves(x_api_key, user_id, gameid, count_most_recent_moves=8))