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


#######API##################


def send_move(move: tuple[int, int], gameId: str):

    # Pre-process args
    move = str(move).strip("()")

    payload = {"teamId": "1397", "move": move, "type": "move", "gameId": gameId}
    headers = {
        "x-api-key": keys.API_KEY,
        "userid": keys.USER_ID,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.post(keys.URL, headers=headers, data=payload)
    print(response.text)


def get_teams():
    payload = {}
    params = {"type": "myTeams"}
    headers = {
        "x-api-key": keys.API_KEY,
        "userid": keys.USER_ID,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.get(keys.URL, headers=headers, data=payload, params=params)

    print(response.text)


def get_board():
    payload = {}
    params = {"type": "boardMap", "gameId": "4671"}
    headers = {
        "x-api-key": keys.API_KEY,
        "userid": keys.USER_ID,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.get(keys.URL, headers=headers, params=params)

    json_board = json.loads(response.text)["output"]

    if json_board == None:
        print("Board is empty")
    else:
        print(json_board)


#######API##################

# get_teams()
# get_board()

# send_move(move=(1, 5), gameId="4671")
