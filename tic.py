import requests
import json

board = []


def min():
    print()


def max():
    print()


def terminal_state():
    print()


#######API##################
url = "https://www.notexponential.com/aip2pgaming/api/index.php"


def send_move():
    print(requests.get())


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


#######API##################

get_teams()
get_board()
