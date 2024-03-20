import requests
import json
import numpy as np

board = np.empty((12, 12), int)


def min():
    print()


def max():
    print()


def terminal_state():
    print()


#######API##################
url = "https://www.notexponential.com/aip2pgaming/api/index.php"


def send_move(move: tuple[int, int], gameId: str):
    
    # Pre-process args
    move = str(move).strip('()')
    
    payload = {
        "teamId": "1397",
        "move": move,
        "type": "move",
        "gameId": gameId
    }
    headers = {
        "x-api-key": "250a442d345be5d375c5",
        "userid": "2620",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.post(url, headers=headers, data=payload)
    print(response.text)


def get_teams():
    payload = {}
    params = {"type": "myTeams"}
    headers = {
        "x-api-key": "250a442d345be5d375c5",
        "userid": "2620",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.get(url, headers=headers, data=payload, params=params)

    print(response.text)


def get_board():
    payload = {}
    params = {"type": "boardMap", "gameId": "4671"}
    headers = {
        "x-api-key": "250a442d345be5d375c5",
        "userid": "2620",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "PostmanRuntime/7.37.0",
    }

    response = requests.get(url, headers=headers, params=params)

    json_board = json.loads(response.text)["output"]

    if json_board == None:
        print("Board is empty")
    else:
        print(json_board)


#######API##################

get_teams()
get_board()

send_move(move=(1, 5), gameId="4671")