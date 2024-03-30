import requests
import json
import numpy as np
import math
from typing import Tuple  # tuple[int, int] notation only works in Python 3.9 and above.
                          # For Python 3.8 and earlier versions, we need to use typing.Tuple
# import keys  #TODO: Still using it?

#########################################
#####            Minimax            #####
#########################################
def minimax(curr_game, state):
    v, move = max_node(curr_game, state)
    return move


def max_node(curr_game, state):
    if curr_game.is_terminal(state):
        return curr_game.utility(state, curr_game.to_move(state)), None

    v = -math.inf
    move = 0
    for successor in curr_game.actions(state):
        v_min, move = min_node(curr_game, curr_game.result(state, successor))
        if v_min > v:
            v, move = v_min, successor

    return v, move


def min_node(curr_game, state):
    if curr_game.is_terminal(state):
        return curr_game.utility(state, curr_game.to_move(state)), None

    v = math.inf
    move = 0
    for successor in curr_game.actions(state):
        v_max, move = max_node(curr_game, curr_game.result(state, successor))
        if v_max < v:
            v, move = v_max, successor

    return v, move



#########################################
#####              API              #####
#####   - API is case sensitive.    #####
#####   - e.g. teamid != teamId     #####
#########################################
URL = "https://www.notexponential.com/aip2pgaming/api/index.php"

#------- One Time Operations -------#
def create_team(x_api_key, user_id, name: str):
    """
    Request Type: POST
    
    Parameters: type=team, name=$name
    Return Values: Team ID.  Fails if team already exists, or team name is too short, or too long.
    """
    
    payload = {"type": "team", "name": name}
    params = {}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.post(URL, headers=headers, data=payload, params=params)
    print(response.text)  # Example Result of Success: {"code":"OK","teamId":1024}
                          # Example Result of Fail: {"code":"FAIL","message":"Invalid name, already exists"}
    response_in_dict = json.loads(response.text)  # Example: {'code': 'OK', 'teamId': 1024}
    
    if response_in_dict["code"] == "OK":
        return response_in_dict["teamId"]  # return Team ID
    elif response_in_dict["code"] == "FAIL":
        print(response_in_dict["message"])
    else:
        print("*** ERROR ***")


def add_team_member(x_api_key, user_id, teamid: str, member_user_id: str):
    """
    Request Type: POST
    
    Parameter: type=member, teamId, userId
    Return Values: OK. Fails if user is already in that team.
    """
    
    payload = {"type": "member", "teamId": teamid, "userId": member_user_id}
    params = {}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.post(URL, headers=headers, data=payload, params=params)
    print(response.text)  # Example Result of Success: {"code":"OK"}
                          # Example Result of Fail: prints nothing.
    if len(response.text) != 0:  # Success
        response_in_dict = json.loads(response.text)
        return response_in_dict["code"]
    else:                        # Fail
        print(f"Fail: '{member_user_id}' is already in the team")


def remove_team_member(x_api_key, user_id, teamid: str, member_user_id: str):
    """
    Request Type: POST
    
    Parameter: type=removeMember, teamId, userId
    Return Values: OK.
    """
    
    payload = {"type": "removeMember", "teamId": teamid, "userId": member_user_id}
    params = {}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.post(URL, headers=headers, data=payload, params=params)
    print(response.text)  # Example Result of Success: {"code":"OK"}
                          # Example Result of Fail: prints nothing.
    if len(response.text) != 0:  # Success / it seems like it's always success.
        response_in_dict = json.loads(response.text)
        return response_in_dict["code"]
    else:                        # Fail
        print(f"Fail: '{member_user_id}' is not in the team")


def get_team_member(x_api_key, user_id, teamid: str):
    """
    Request Type: GET
    
    Parameters: type=team, teamId=$teamid
    Return Values: userids, comma separated
    """
    
    payload = {}
    params = {"type": "team", "teamId": teamid}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.get(URL, headers=headers, data=payload, params=params)
    print(response.text)  # Example Result of Success: {"userIds":["79","85","86","87","88"],"code":"OK"}
                          # Example Result of Fail: {"code":"FAIL","message":"No team (or no members!) for team: 0"}
    response_in_dict = json.loads(response.text)  # Example: {'userIds': ['79', '85', '86', '87', '88'], 'code': 'OK'}
    
    if response_in_dict["code"] == "OK":
        comma_separated_members = ",".join(response_in_dict["userIds"])
        return comma_separated_members  # userids, comma separated
    elif response_in_dict["code"] == "FAIL":
        print(response_in_dict["message"])
    else:
        print("*** ERROR ***")


def get_my_teams(x_api_key, user_id):
    """
    Request Type: GET
    
    Parameters: type=myTeams
    Return Values: teams, comma separated.  Generally, this should be just one.

    https://www.notexponential.com/aip2pgaming/api/index.php?type=myTeams
    """
    
    payload = {}
    params = {"type": "myTeams"}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.get(URL, headers=headers, data=payload, params=params)
    print(response.text)  # Example Result of Success: {"myTeams":[{"1397":"LTE"}],"code":"OK"}
                          # Example Result of Fail: {"code":"FAIL","message":"Invalid API Key + userId combination"}
    response_in_dict = json.loads(response.text)  # Example: {'myTeams': [{'1397': 'LTE'}, {'1416': '5G_UWB'}], 'code': 'OK'}
    
    if response_in_dict["code"] == "OK":
        my_team_list = [list(dictionary.keys())[0] for dictionary in response_in_dict["myTeams"]]
        comma_separated_teams = ",".join(my_team_list)
        return comma_separated_teams  # teams, comma separated
    elif response_in_dict["code"] == "FAIL":
        print(response_in_dict["message"])
    else:
        print("*** ERROR ***")


#------- Playing Games / Ongoing Operations -------#
def create_game(x_api_key, user_id, teamid1: str, teamid2: str, board_size=20, target_num=10):
    """
    Request Type: POST
    
    Parameters: 
    - type=game
    - teamId1
    - teamId2
    - gameType=TTT  // This is the only value supported this semester
    - Optionally:
        - boardSize=20
        - target=10 (Needs to be <= boardSize) 
        // Default values are 12 and 6
    Return Values: GameID
    """

    payload = {"type": "game", "teamId1": teamid1, "teamId2": teamid2, "gameType": "TTT", "boardSize": board_size, "target": target_num}
    params = {}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.post(URL, headers=headers, data=payload, params=params)
    print(response.text)  # Example Result of Success: {"code":"OK","gameId":1290}
                          # Example Result of Fail: {"code":"FAIL","message":"Target (12) cannot exceed board size: 10"}
    response_in_dict = json.loads(response.text)  # Example: {'code': 'OK', 'gameId': 1290}
    
    if response_in_dict["code"] == "OK":
        gameId = response_in_dict["gameId"]
        return gameId   # return GameID
    elif response_in_dict["code"] == "FAIL":
        print(response_in_dict["message"])  # Example: Target (12) cannot exceed board size: 10
    else:
        print("*** ERROR ***")


def get_my_games(x_api_key, user_id, history_type: str):
    """
    Request Type: GET
    
    Parameters: type=myGames or myOpenGames
    ## myGames: Shows you every game you played including the closed ones.
    ## myOpenGames: Shows you the only 'open' games that is not ended yet.
    Return Values:games, comma separated
    """
    
    payload = {}
    params = {"type": history_type}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.get(URL, headers=headers, data=payload, params=params)
    print(response.text)  # Example Result of Success: {"myGames":[{"4671":"1397:1397:C:1397"},{"4749":"1397:1397:O"},{"4751":"1416:1397:O"}],"code":"OK"}
                          # Example Result of Fail: {"code":"FAIL","message":"Invalid API Key + userId combination"}
    response_in_dict = json.loads(response.text)  # Example: {'myGames': [{'4671': '1397:1397:C:1397'}, {'4749': '1397:1397:O'}], 'code': 'OK'}
    
    if response_in_dict["code"] == "OK":
        my_game_list = [list(dictionary.keys())[0] for dictionary in response_in_dict["myGames"]]
        games = ",".join(my_game_list)
        return games   # games, comma separated
    elif response_in_dict["code"] == "FAIL":
        print(response_in_dict["message"])  # Example: Invalid API Key + userId combination
    else:
        print("*** ERROR ***")
    

def make_move(x_api_key, user_id, game_id: str, team_id: str, where_to_move: Tuple[int, int]):
    """
    Request Type: POST
    
    Parameter: type=move, gameId, teamId, move
    Return Value: Move ID. 
    Fails in following cases:
    - If no such game
    - If the team is not a participant in the game
    - If it is not the move of the team making that move
    - If the move dimensions are negative or >= n.  (Move starts from 0,0.  That is, 0 - indexing]
    """

    # Pre-process args
    where_to_move = str(where_to_move).strip("()")

    payload = {"type": "move", "gameId": game_id, "teamId": team_id, "move": where_to_move}
    params = {}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.post(URL, headers=headers, data=payload, params=params)
    print(response.text)  # Example Result of Success: {"moveId":105862,"code":"OK"}
                          # Example Result of Fail when it's not your team's turn: {"code":"FAIL","message":"Cannot make move - It is not the turn of team: 1397"}
                          # Example Result of Fail when tried to move to where it's already visited: it prints out an empty space.
    
    if len(response.text) != 0:  # There is some messages in the response
        response_in_dict = json.loads(response.text)  # Example: {'moveId': 105862, 'code': 'OK'}
        if response_in_dict["code"] == "OK":      # Success
            return response_in_dict["moveId"]
        elif response_in_dict["code"] == "FAIL":  # Fail
            print(response_in_dict["message"])    # Example: Cannot make move - It is not the turn of team: 1397
        else:
            print("*** ERROR ***")
    else:  # Fail
        print("Fails in following cases: ")
        print("- If no such game")
        print("- If the team is not a participant in the game")
        print("- If it is not the move of the team making that move")
        print("- If the move dimensions are negative or >= n.  (Move starts from 0,0.  That is, 0 - indexing)")


def get_moves():
    """
    Request Type: GET
    
    Parameters: type=moves, gameId, Count of most recent of moves
    Return Values: List of Moves, comma separated
    """
    pass

def get_game_details():
    """
    Request Type: GET
    
    Parameters: type=gameDetails, gameId
    Return Values: Game Details as JSON
    """
    pass

def get_board_string():
    """
    Request Type: GET
    
    Parameters: type=boardString, gameId
    Return Values: Board, in form of a string of O,X,-
    """
    pass
    
# def get_board_map(x_api_key, user_id):
#     """
#     Request Type: GET
    
#     Parameters: type=boardMap, gameId
#     Return Values: Board, in form of a string of O,X,-
#     """
    
#     payload = {}
#     params = {"type": "boardMap", "gameId": "4671"}
#     headers = {
#         "x-api-key": x_api_key,
#         "userId": user_id,
#         "Content-Type": "application/x-www-form-urlencoded",
#         "User-Agent": "PostmanRuntime/7.37.0",
#     }

#     response = requests.get(URL, headers=headers, params=params)

#     json_board = json.loads(response.text)["output"]

#     if json_board == None:
#         print("Board is empty")
#     else:
#         print(json_board)



################## for Testing ##################

#TODO: Make sure to delete the api-keys
x_api_key = None
user_id = None
teamid = None
teamid2 = None  # 5G_UWB
gameid = None

###------- One Time Operations -------###
# create_team(x_api_key, user_id, name="5G_UWB")
# add_team_member(x_api_key, user_id, teamid, member_user_id="2638")
# remove_team_member(x_api_key, user_id, teamid, member_user_id="2638")
# get_team_member(x_api_key, user_id, teamid="1416")
# get_my_teams(x_api_key, user_id)

###------- Playing Games / Ongoing Operations -------###
# create_game(x_api_key, user_id, teamid2, teamid)                                # Success example
# create_game(x_api_key, user_id, teamid2, teamid, board_size=10, target_num=12)  # Fail example (Because target_num is bigger than the board_size)
# get_my_games(x_api_key, user_id, history_type="myGames")      # Every game you've played
# get_my_games(x_api_key, user_id, history_type="myOpenGames")  # Only Opened games
make_move(x_api_key, user_id, gameid, teamid2, where_to_move=(9,0))


# get_my_teams()
# get_board_map(x_api_key, user_id)
