import pandas as pd
import requests
import io
import os
from typing import List, Dict, Any
import time
import concurrent.futures

# Add proper chess module import with error handling
try:
    import chess.pgn
except ImportError:
    raise ImportError(
        "python-chess is required. Install it with: pip install python-chess"
    )

def getGames(user: str) -> List[Dict[str, Any]]:
    """Get all games with optimized API requests"""
    headers = {
        'User-Agent': 'Chess Analysis App (Contact: github.com/yourusername)',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip'
    }
    
    try:
        # Use session for connection pooling
        with requests.Session() as session:
            session.headers.update(headers)
            
            # Get archives with proper caching
            archives_url = f"https://api.chess.com/pub/player/{user}/games/archives"
            archives_response = session.get(archives_url)
            archives_response.raise_for_status()
            
            archives_data = archives_response.json()
            if "archives" not in archives_data:
                raise Exception("No game archives found")
            
            # Fetch archives in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {
                    executor.submit(session.get, url): url 
                    for url in archives_data["archives"][-12:]  # Last 12 months only
                }
                
                all_games = []
                for future in concurrent.futures.as_completed(future_to_url):
                    try:
                        response = future.result()
                        response.raise_for_status()
                        monthly_games = response.json()
                        if "games" in monthly_games:
                            all_games.extend(monthly_games["games"])
                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching games: {str(e)}")
                        continue
                        
            return all_games
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error accessing chess.com API: {str(e)}")

def createDataset(li: List[Dict[str, Any]], user: str) -> None:
    """Create a dataset from chess games and save to CSV"""
    col = [
        'player_username', 'opponent_username', 'played_as', 'opponent_played_as',
        'result_for_player', 'result_for_opponent', 'player_rating', 'opponent_rating',
        'time_class', 'opening', 'moves', 'first_move', 'rated', 'PGN', 'FEN'
    ]

    df = pd.DataFrame(columns=col)

    for x in li:
        try:
            liz = [None] * 15

            if x["rules"] != "chess":
                continue

            liz[0] = user
            liz[8] = x["time_class"]
            liz[13] = x["pgn"]
            liz[14] = x["fen"]
            liz[12] = x["rated"]

            try:
                pgn = chess.pgn.read_game(io.StringIO(x["pgn"]))
                if pgn and "ECOUrl" in pgn.headers:
                    opening = pgn.headers["ECOUrl"][31:].replace("-", " ")
                    liz[9] = opening
            except Exception as e:
                print(f"Warning: Could not parse PGN for game: {str(e)}")
                continue

            count = 0
            for moves in pgn.mainline_moves():
                if count == 0:
                    liz[11] = str(moves)
                count += 1

            liz[10] = str(int(count / 2))

            if x["white"]["username"] == user:
                liz[2] = "white"
                liz[3] = "black"
                liz[4] = x["white"]["result"]
                liz[5] = x["black"]["result"]
                liz[6] = x["white"]["rating"]
                liz[7] = x["black"]["rating"]
                liz[1] = x["black"]["username"]
            else:
                liz[2] = "black"
                liz[3] = "white"
                liz[4] = x["black"]["result"]
                liz[5] = x["white"]["result"]
                liz[6] = x["black"]["rating"]
                liz[7] = x["white"]["rating"]
                liz[1] = x["white"]["username"]

            if None not in liz:
                df.loc[len(df)] = liz
                
        except Exception as e:
            print(f"Warning: Could not process game: {str(e)}")
            continue

    # Create directory if it doesn't exist
    os.makedirs(user, exist_ok=True)
    df.to_csv(os.path.join(user, 'chess_dataset.csv'), index=False)

def driver_fn(username: str) -> None:
    """Main driver function to get and process chess.com data"""
    try:
        # Get all games
        all_games_List = getGames(username)
        if not all_games_List:
            raise Exception("No games found for this user")
            
        # Create dataset
        createDataset(all_games_List, username)
        
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")