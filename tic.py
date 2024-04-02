import requests
import json
import numpy as np
import math
from typing import Tuple  # tuple[int, int] notation only works in Python 3.9 and above.
import keys

# For Python 3.8 and earlier versions, we need to use typing.Tuple
# import keys  #TODO: Still using it?


#########################################
#####            Minimax            #####
#########################################

## Killer move ordering, tranposition table


def minimax(curr_game, state, depth=4):
    v, move = max_node(curr_game, state, -math.inf, math.inf, depth)
    print(f'in minimax, v={v}, move={move}')
    return move


def max_node(curr_game, state, alpha, beta, depth):

    if depth == 0:
        v = curr_game.eval(state, curr_game.to_move(state))
        return v, None

    if curr_game.is_terminal(state):
        v = curr_game.eval(state, curr_game.to_move(state))
        return v, None

    # if curr_game.is_cutoff(state, depth):
    #     v = curr_game.eval(state, curr_game.to_move(state))
    #     # print(f"in max, v={v}, state=\n{state.state}")
    #     return v, None
 

    v = -math.inf
    for successor in curr_game.actions(state):
        v_min, min_move = min_node(
            curr_game, curr_game.result(state, successor), alpha, beta, depth - 1
        )
        if v_min > v:
            # print(f"in max, b/c {v_min} > {v}, switched move v_min={v_min} move={successor}")
            v, move = v_min, successor
            alpha = max(alpha, v)
        if v >= beta:
            return v, move

    return v, move


def min_node(curr_game, state, alpha, beta, depth):

    if depth == 0:
        v = curr_game.eval(state, curr_game.to_move(state))
        return v, None

    if curr_game.is_terminal(state):
        v = curr_game.eval(state, curr_game.to_move(state))
        return v, None
    
    # if curr_game.is_cutoff(state, depth):
    #     v = curr_game.eval(state, curr_game.to_move(state))
    #     return v, None

    v = math.inf
    for successor in curr_game.actions(state):
        v_max, max_move = max_node(
            curr_game, curr_game.result(state, successor), alpha, beta, depth - 1
        )
        if v_max < v:
            v, move = v_max, successor
            beta = min(beta, v)
        if v <= alpha:
            return v, move

    return v, move


#########################################
#####              API              #####
#####   - API is case sensitive.    #####
#####   - e.g. teamid != teamId     #####
#########################################
URL = "https://www.notexponential.com/aip2pgaming/api/index.php"


# ------- One Time Operations -------#
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
    response_in_dict = json.loads(
        response.text
    )  # Example: {'code': 'OK', 'teamId': 1024}

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
    else:  # Fail
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
    else:  # Fail
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
    print(
        response.text
    )  # Example Result of Success: {"userIds":["79","85","86","87","88"],"code":"OK"}
    # Example Result of Fail: {"code":"FAIL","message":"No team (or no members!) for team: 0"}
    response_in_dict = json.loads(
        response.text
    )  # Example: {'userIds': ['79', '85', '86', '87', '88'], 'code': 'OK'}

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
    print(
        response.text
    )  # Example Result of Success: {"myTeams":[{"1397":"LTE"}],"code":"OK"}
    # Example Result of Fail: {"code":"FAIL","message":"Invalid API Key + userId combination"}
    response_in_dict = json.loads(
        response.text
    )  # Example: {'myTeams': [{'1397': 'LTE'}, {'1416': '5G_UWB'}], 'code': 'OK'}

    if response_in_dict["code"] == "OK":
        my_team_list = [
            list(dictionary.keys())[0] for dictionary in response_in_dict["myTeams"]
        ]  # TODO
        comma_separated_teams = ",".join(my_team_list)
        return comma_separated_teams  # teams, comma separated
    elif response_in_dict["code"] == "FAIL":
        print(response_in_dict["message"])
    else:
        print("*** ERROR ***")


# ------- Playing Games / Ongoing Operations -------#
def create_game(
    x_api_key, user_id, teamid1: str, teamid2: str, board_size=20, target_num=10
):
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

    payload = {
        "type": "game",
        "teamId1": teamid1,
        "teamId2": teamid2,
        "gameType": "TTT",
        "boardSize": board_size,
        "target": target_num,
    }
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
    response_in_dict = json.loads(
        response.text
    )  # Example: {'code': 'OK', 'gameId': 1290}

    if response_in_dict["code"] == "OK":
        gameId = response_in_dict["gameId"]
        return gameId  # return GameID
    elif response_in_dict["code"] == "FAIL":
        print(
            response_in_dict["message"]
        )  # Example: Target (12) cannot exceed board size: 10
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
    print(
        response.text
    )  # Example Result of Success: {"myGames":[{"4671":"1397:1397:C:1397"},{"4749":"1397:1397:O"},{"4751":"1416:1397:O"}],"code":"OK"}
    # Example Result of Fail: {"code":"FAIL","message":"Invalid API Key + userId combination"}
    response_in_dict = json.loads(
        response.text
    )  # Example: {'myGames': [{'4671': '1397:1397:C:1397'}, {'4749': '1397:1397:O'}], 'code': 'OK'}

    if response_in_dict["code"] == "OK":
        my_game_list = [
            list(dictionary.keys())[0] for dictionary in response_in_dict["myGames"]
        ]  # TODO
        # comma_separated = ','.join([str(game) for game in my_game_list])
        games = ",".join(my_game_list)
        return games  # games, comma separated -- It's returning gameIds
    elif response_in_dict["code"] == "FAIL":
        print(
            response_in_dict["message"]
        )  # Example: Invalid API Key + userId combination
    else:
        print("*** ERROR ***")


def make_move(
    x_api_key, user_id, game_id: str, team_id: str, where_to_move: Tuple[int, int]
):
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

    payload = {
        "type": "move",
        "gameId": game_id,
        "teamId": team_id,
        "move": where_to_move,
    }
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
        response_in_dict = json.loads(
            response.text
        )  # Example: {'moveId': 105862, 'code': 'OK'}
        if response_in_dict["code"] == "OK":  # Success
            return response_in_dict["moveId"]
        elif response_in_dict["code"] == "FAIL":  # Fail
            print(
                response_in_dict["message"]
            )  # Example: Cannot make move - It is not the turn of team: 1397
        else:
            print("*** ERROR ***")
    else:  # Fail
        print("Fails in following cases: ")
        print("- If no such game")
        print("- If the team is not a participant in the game")
        print("- If it is not the move of the team making that move")
        print(
            "- If the move dimensions are negative or >= n.  (Move starts from 0,0.  That is, 0 - indexing)"
        )


def get_moves(x_api_key, user_id, game_id, count_most_recent_moves):
    """
    Request Type: GET

    Parameters: type=moves, gameId, Count of most recent of moves
    Return Values: List of Moves, comma separated
    """

    payload = {}
    params = {"type": "moves", "gameId": game_id, "count": count_most_recent_moves}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.get(URL, headers=headers, data=payload, params=params)
    # print(response.text)  # Example Result of Success: {"moves":[{"moveId":"105889","gameId":"4751","teamId":"1416","move":"9,0","symbol":"O","moveX":"9","moveY":"0"},{"moveId":"105888","gameId":"4751","teamId":"1397","move":"8,10","symbol":"X","moveX":"8","moveY":"10"}],"code":"OK"}
    # Example Result of Fail: {"code":"FAIL","message":"No moves"}
    response_in_dict = json.loads(
        response.text
    )  # Example: {"moves": [{"moveId": "105889", "gameId": "4751", "teamId": "1416", "move": "9,0", "symbol": "O", "moveX": "9", "moveY": "0"}], "code": "OK"}

    if response_in_dict["code"] == "OK":
        # my_move_list = [list(dictionary.values())[0] for dictionary in response_in_dict["moves"]]
        my_move_list = [dictionary for dictionary in response_in_dict["moves"]]
        # print(my_move_list)
        # list_of_moves = ",".join(my_move_list)
        return my_move_list  # List of moves, comma separated
    elif response_in_dict["code"] == "FAIL":
        print(response_in_dict["message"])  # Example: No moves
    else:
        print("*** ERROR ***")


def get_game_details(x_api_key, user_id, game_id):
    """
    Request Type: GET

    Parameters: type=gameDetails, gameId
    Return Values: Game Details as JSON
    """
    # list_from_s = json.loads(s.replace("'", '"'))

    payload = {}
    params = {"type": "gameDetails", "gameId": game_id}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.get(URL, headers=headers, data=payload, params=params)
    # print(response.text)  # Example Result of Success: {"game":"{\"gameid\":\"4750\",\"gametype\":\"TTT\",\"moves\":\"0\",\"boardsize\":\"20\",\"target\":\"10\",\"team1id\":\"1416\",\"team1Name\":\"5G_UWB\",\"team2id\":\"1397\",\"team2Name\":\"LTE\",\"secondspermove\":\"600\",\"status\":\"O\",\"winnerteamid\":null,\"turnteamid\":\"1416\"}","code":"OK"}
    #                       # Example Result of Success, but actually a Fail: {"game":"{}","code":"OK"}
    response_in_dict = json.loads(
        response.text
    )  # Example: {"game": "{\"gameid\":\"4750\",\"gametype\":\"TTT\",\"moves\":\"0\",\"boardsize\":\"20\",\"target\":\"10\",\"team1id\":\"1416\",\"team1Name\":\"5G_UWB\",\"team2id\":\"1397\",\"team2Name\":\"LTE\",\"secondspermove\":\"600\",\"status\":\"O\",\"winnerteamid\":null,\"turnteamid\":\"1416\"}","code":"OK"}

    if response_in_dict["code"] == "OK":
        if len(response_in_dict["game"]) != 0:
            game_details = response_in_dict["game"]
            game_details_in_dict = json.loads(game_details)
            # print(game_details_in_dict)
            return game_details_in_dict  # return Game Details as JSON
        elif len(response_in_dict["game"]) == 0:
            game_details = response_in_dict["game"]
            game_details_in_dict = json.loads(game_details)
            # print(game_details_in_dict)
            return game_details_in_dict  # return Game Details as JSON
    elif response_in_dict["code"] == "FAIL":
        print(response_in_dict["message"])
    else:
        print("*** ERROR ***")


def get_board_string(x_api_key, user_id, game_id):
    """
    Request Type: GET

    Parameters: type=boardString, gameId
    Return Values: Board, in form of a string of O,X,-
    """

    payload = {}
    params = {"type": "boardString", "gameId": game_id}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.get(URL, headers=headers, data=payload, params=params)
    print(
        response.text
    )  # Example Result of Success: {"output":"OX------------------\nO-------------------\nO-------------------\nOX------------------\nO-X-----------------\nOX------------------\nO-----X-------------\nO------X-X----------\nO-------X-X---------\nO-------------------\n--------------------\n--------------------\n--------------------\n--------------------\n--------------------\n--------------------\n--------------------\n--------------------\n--------------------\n--------------------\n","target":10,"code":"OK"}
    # Example Result of Fail: {"code":"FAIL","message":"Invalid game ID"}
    response_in_dict = json.loads(
        response.text
    )  # Example: {"moves": [{"moveId": "105889", "gameId": "4751", "teamId": "1416", "move": "9,0", "symbol": "O", "moveX": "9", "moveY": "0"}], "code": "OK"}

    if response_in_dict["code"] == "OK":
        board = response_in_dict["output"]
        print(f"[GAMEBOARD for Game#{game_id}]")
        rows = board.strip().split("\n")
        board_2d = [list(row) for row in rows]
        # print(board_2d)
        [print(i) for i in board_2d]
        return board_2d  # return boardString
    elif response_in_dict["code"] == "FAIL":
        print(
            response_in_dict["message"]
        )  # Example: Target (12) cannot exceed board size: 10
    else:
        print("*** ERROR ***")


def get_board_map(x_api_key, user_id, game_id):
    """
    Request Type: GET

    Parameters: type=boardMap, gameId
    Return Values: Board, in form of a string of O,X,-
    """

    payload = {}
    params = {"type": "boardMap", "gameId": game_id}
    headers = {
        "x-api-key": x_api_key,
        "userId": user_id,
        "Content-Type": "applicati  on/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.get(URL, headers=headers, data=payload, params=params)
    # print(response.text)  # Example Result of Success: {"output":"{\"0,0\":\"O\",\"0,1\":\"X\",\"1,0\":\"O\",\"3,1\":\"X\",\"2,0\":\"O\",\"4,2\":\"X\",\"3,0\":\"O\",\"5,1\":\"X\",\"4,0\":\"O\",\"8,8\":\"X\",\"5,0\":\"O\",\"7,7\":\"X\",\"6,0\":\"O\",\"6,6\":\"X\",\"7,0\":\"O\",\"7,9\":\"X\",\"8,0\":\"O\",\"8,10\":\"X\",\"9,0\":\"O\"}","target":10,"code":"OK"}
    #                       # Example Result of Fail: {"code":"FAIL","message":"Invalid game ID"}
    response_in_dict = json.loads(
        response.text
    )  # Example: {"moves": [{"moveId": "105889", "gameId": "4751", "teamId": "1416", "move": "9,0", "symbol": "O", "moveX": "9", "moveY": "0"}], "code": "OK"}

    if response_in_dict["code"] == "OK":
        board = response_in_dict["output"]
        board_in_dict = json.loads(board)
        # print(f"[Where tiles are places for Game#{game_id}]")
        # print(board_in_dict)
        return board_in_dict  # return boardMap
    elif response_in_dict["code"] == "FAIL":
        print(response_in_dict["message"])  # Example: Invalid game ID
    else:
        print("*** ERROR ***")


################## for Testing ##################

# TODO: Make sure to delete the api-keys
x_api_key = keys.API_KEY  # Your API-KEY
user_id = keys.USER_ID  # Your ID
teamid = keys.TEAM_ID  # Your Team ID
# teamid2 = "1416"  # Enemy Team ID, 5G_UWB
# gameid = "4751"   # game ID you are playing
gameid = "4783"

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
# make_move(x_api_key, user_id, gameid, teamid, where_to_move=(2,2))
# print(get_moves(x_api_key, user_id, gameid, count_most_recent_moves="1"))
# print(get_game_details(x_api_key, user_id, gameid))
# get_board_string(x_api_key, user_id, gameid)
# get_board_map(x_api_key, user_id, gameid)


# get_my_teams()
# get_board_map(x_api_key, user_id)
