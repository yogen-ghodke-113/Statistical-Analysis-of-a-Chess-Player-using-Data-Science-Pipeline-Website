import requests


def getGames(user):
    req = "https://api.chess.com/pub/player/" + user + "/games/archives"
    li = requests.get(req).json()["archives"]
    li_games = []
    for x in li:
        li_games.append(requests.get(x).json())
    all_games = []
    for x in li_games:
        all_games.extend(x["games"])
    return all_games


def createDataset(li, user):
    import pandas as pd
    import chess.pgn
    import io

    di = {
        'player_username': [],
        'opponent_username': [],
        'played_as': [],
        'opponent_played_as': [],
        'result_for_player': [],
        'result_for_opponent': [],
        'player_rating': [],
        'opponent_rating': [],
        'time_class': [],
        'opening': [],
        'moves': [],
        'first_move': [],
        'rated': [],
        'PGN': [],
        'FEN': []
    }

    for x in li:

        if x["rules"] != "chess":
            continue

        di['player_username'].append(user)
        di['time_class'].append(x["time_class"])
        di['PGN'].append(x["pgn"])
        di['FEN'].append(x["fen"])
        di['rated'].append(x["rated"])

        pgn = chess.pgn.read_game(io.StringIO(x["pgn"]))

        try:
            opening = pgn.headers["ECOUrl"][31:].replace("-", " ")
            di['opening'].append(opening)
        except:
            di['opening'].append("Nan")

        count = 0
        for moves in pgn.mainline_moves():
            if count == 0:
                di['first_move'].append(str(moves))
            count += 1

        di['moves'].append(str(int(count / 2)))

        if x["white"]["username"] == user:
            di['played_as'].append("white")
            di["opponent_played_as"].append("black")
            di["result_for_player"].append(x["white"]["result"])
            di["result_for_opponent"].append(x["black"]["result"])
            di["player_rating"].append(x["white"]["rating"])
            di["opponent_rating"].append(x["black"]["rating"])
            di['opponent_username'].append(x["black"]["username"])

        else:
            di['opponent_username'].append(x["white"]["username"])
            di["played_as"].append("black")
            di["opponent_played_as"].append("white")
            di["result_for_player"].append(x["black"]["result"])
            di["result_for_opponent"].append(x["white"]["result"])
            di["player_rating"].append(x["black"]["rating"])
            di["opponent_rating"].append(x["white"]["rating"])

    df = pd.DataFrame(di)
    index_names = df[df['opening'] == "Nan"].index
    df.drop(index_names, inplace=True)
    df.to_csv(user + '/chess_dataset.csv', index=False)


def driver_fn(username):
    all_games_List = getGames(username)

    import os
    path = os.path.join("", username)
    try:
        os.mkdir(path)
    except:
        pass
    createDataset(all_games_List, username)
