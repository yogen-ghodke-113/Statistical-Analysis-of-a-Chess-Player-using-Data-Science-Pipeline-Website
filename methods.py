import json
import requests
import tree as tr
import chess.pgn
import io


# Returns list of dictionaries received from server
def getAPI(query):
    queries = []
    req = "https://api.chess.com/pub/player/" + query
    queries.append(requests.get(req).json())
    return queries


# Returns list of dictionaries containing all games ever played.
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


# Prints list of dictionaries in a formatted manner
def display(li):
    for x in li:
        print(json.dumps(x, indent=4))


# Filter to only classical games
def filterList(li, user):
    for x in li:
        if x["rules"] != "chess":
            li.remove(x)
            continue

        x["black"].pop("rating")
        x["white"].pop("@id")
        if x["white"]["username"] == user:
            x.update({"color": "white", "result": x["white"]["result"]})
        else:
            x.update({"color": "black", "result": x["black"]["result"]})

        j = ["url", "rated", "time_control", "end_time", "fen", "time_class", "rules", "white", "black"]
        for k in j:
            x.pop(k)

def createDataset(li,user):
    pass


# x = name of opening, li = List of all the moves made in that opening
def buildOpeningTree(openings):
    tree = tr.Tree()
    for x in openings:
        li = openings[x].split(" ")
        tree.builder(tree.root, x, li)
    return tree


def traverseToNode(trie):
    s = "e2e4"
    p = s.split(" ")
    q = trie.traverse(p, trie.root)


def otherMethodCalls():
    # query1 = getAPI(username)
    # query2 = getAPI(username+"/stats")
    # display([query1,query2])
    pass


def convertPGN(games, white, black, w_freq, b_freq):
    for x in games:
        try:
            pgn = io.StringIO(x["pgn"])
            game = chess.pgn.read_game(pgn)
            li = []
            for move in game.mainline_moves():
                li.append(str(move))

                if x["color"] == "white":
                    white.insertGames(li, white.root, x["result"], w_freq)
                else:
                    black.insertGames(li, black.root, x["result"], b_freq)
        except:
            continue

# def posnFreq():

# def findFavOpening(openings, trie, )
