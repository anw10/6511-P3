import requests
import json
import numpy as np
import math
import keys

board = np.empty((12, 12), int)


def minimax():
    move = max(board)
    return move


def max(board):
    v = -math.inf
    print()


def min(board):
    v = math.inf
    print()


def terminal_state():
    print()


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

get_teams()
get_board()

send_move(move=(1, 5), gameId="4671")
