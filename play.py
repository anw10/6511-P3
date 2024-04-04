import game
import keys
from tic import *

################## For Playing ##################

###------- Keys -------###

x_api_key = keys.API_KEY    # Your API-KEY
user_id = keys.USER_ID      # Your ID
teamid = keys.TEAM_ID       # Your Team ID
# teamid2 = ""              # Opponent Team ID
# gameid = ""               # Current game

###------- Start Game -------###

# GTTT = game.Game(n=10, target=5, DEBUG_STATE=None, DEBUG_PRINT=False)                             # Initialize n and m
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
# create_game(x_api_key, user_id, teamid, teamid2, board_size=5, target_num=4)                                # Success example
# create_game(x_api_key, user_id, teamid2, teamid, board_size=10, target_num=12)  # Fail example (Because target_num is bigger than the board_size)
# get_my_games(x_api_key, user_id, history_type="myGames")      # Every game you've played
# get_my_games(x_api_key, user_id, history_type="myOpenGames")  # Only Opened games
# make_move(x_api_key, user_id, gameid, teamid, where_to_move=(2,2))

# print(get_game_details(x_api_key, user_id, gameid))
# get_board_string(x_api_key, user_id, gameid)
# get_board_map(x_api_key, user_id, gameid)
# print(get_game_details(x_api_key, user_id, gameid))
# get_my_teams()
# get_board_map(x_api_key, user_id)