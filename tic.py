import requests
import json
import numpy as np
import math
import keys


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
def create_team():
    """
    Request Type: POST
    
    Parameters: type=team, name=$name
    Return Values: Team ID.  Fails if team already exists, or team name is too short, or too long.
    """    
    pass

def add_team_member():
    """
    Request Type: POST
    
    Parameter: type=member, teamId, userId
    Return Values: OK. Fails if user is already in that team.
    """
    pass

def remove_team_member():
    """
    Request Type: POST
    
    Parameter: type=removeMember, teamId, userId
    Return Values: OK.
    """
    pass

def get_team_member():
    """
    Request Type: GET
    
    Parameters: type=team, teamId=$teamid
    Return Values: userids, comma separated
    """
    pass

def get_my_teams():
    """
    Request Type: GET
    
    Parameters: type=myTeams
    Return Values: teams, comma separated.  Generally, this should be just one.

    https://www.notexponential.com/aip2pgaming/api/index.php?type=myTeams
    """
    payload = {}
    params = {"type": "myTeams"}
    headers = {
        "x-api-key": keys.API_KEY,
        "userid": keys.USER_ID,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    # response = requests.get(keys.URL, headers=headers, data=payload, params=params)
    response = requests.get(URL, headers=headers, data=payload, params=params)

    print(response.text)

#------- Playing Games / Ongoing Operations -------#
def create_game():
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
    pass

def get_my_games():
    """
    Request Type: GET
    
    Parameters: type=myGames or myOpenGames 
    Return Values:games, comma separated
    """
    pass

def make_move(move: tuple[int, int], gameId: str):
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
    move = str(move).strip("()")

    payload = {"teamId": "1397", "move": move, "type": "move", "gameId": gameId}
    headers = {
        "x-api-key": keys.API_KEY,
        "userid": keys.USER_ID,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    # response = requests.post(keys.URL, headers=headers, data=payload)
    response = requests.post(URL, headers=headers, data=payload)
    print(response.text)

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
    
def get_board_map():
    """
    Request Type: GET
    
    Parameters: type=boardMap, gameId
    Return Values: Board, in form of a string of O,X,-
    """
    pass

# def get_board():
#     payload = {}
#     params = {"type": "boardMap", "gameId": "4671"}
#     headers = {
#         "x-api-key": keys.API_KEY,
#         "userid": keys.USER_ID,
#         "Content-Type": "application/x-www-form-urlencoded",
#         "User-Agent": "PostmanRuntime/7.37.0",
#     }

#     # response = requests.get(keys.URL, headers=headers, params=params)
#     response = requests.get(URL, headers=headers, params=params)

#     json_board = json.loads(response.text)["output"]

#     if json_board == None:
#         print("Board is empty")
#     else:
#         print(json_board)


################## for Testing ################## #TODO: cleanup when it's done

# get_my_teams()
# get_board()

# make_move(move=(1, 5), gameId="4671")
