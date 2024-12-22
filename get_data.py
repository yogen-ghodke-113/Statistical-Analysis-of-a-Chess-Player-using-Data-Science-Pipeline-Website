import pandas as pd
import requests
import io
import os
from typing import List, Dict, Any
import time
import concurrent.futures
import logging

# Add proper chess module import with error handling
try:
    import chess.pgn
except ImportError:
    raise ImportError(
        "python-chess is required. Install it with: pip install python-chess"
    )

logger = logging.getLogger(__name__)

def getGames(username: str) -> List[Dict[str, Any]]:
    """Get all games for a user from chess.com API"""
    headers = {
        'User-Agent': 'Chess Analysis App (Contact: github.com/yogen-ghodke-113)',
        'Accept': 'application/json'
    }
    
    try:
        # First get archives list
        archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
        response = requests.get(archives_url, headers=headers, timeout=10)
        
        if response.status_code == 404:
            raise Exception("User not found. Please check the username and try again.")
        response.raise_for_status()
        
        archives = response.json()["archives"]
        
        # Get games from all archives
        all_games = []
        for archive_url in archives:  # Get all archives instead of just last 6 months
            try:
                response = requests.get(archive_url, headers=headers, timeout=10)
                response.raise_for_status()
                all_games.extend(response.json()["games"])
                logger.info(f"Fetched {len(response.json()['games'])} games from {archive_url}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Could not fetch archive {archive_url}: {str(e)}")
                continue
        
        if not all_games:
            raise Exception("No games found for this user")
        
        logger.info(f"Total games fetched: {len(all_games)}")
        return all_games
        
    except requests.exceptions.RequestException as e:
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
            raise Exception("User not found. Please check the username and try again.")
        elif "getaddrinfo failed" in str(e):
            raise Exception("Unable to connect to Chess.com. Please check your internet connection.")
        else:
            raise Exception(f"Error accessing Chess.com API: {str(e)}")

def filterList(games: List[Dict[str, Any]], username: str) -> None:
    """Filter games list to keep only standard chess games"""
    # Remove non-standard chess games (variants, etc.)
    for game in games[:]:  # Create a copy to iterate while modifying
        if game.get("rules") != "chess":
            games.remove(game)
            continue
        
        # Remove games without proper PGN
        if not game.get("pgn"):
            games.remove(game)
            continue
        
        # Remove games where the user didn't play
        if (game.get("white", {}).get("username") != username and 
            game.get("black", {}).get("username") != username):
            games.remove(game)

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

    # Create directory in player_data if it doesn't exist
    user_dir = os.path.join('player_data', user)
    os.makedirs(user_dir, exist_ok=True)
    df.to_csv(os.path.join(user_dir, 'chess_dataset.csv'), index=False)

def createAdvancedDataset(username: str) -> None:
    """Create advanced dataset with numerical result values for prediction"""
    try:
        # Read the original dataset
        df = pd.read_csv(os.path.join('player_data', username, 'chess_dataset.csv'))
        
        # Create result value column (0 for loss, 0.5 for draw, 1 for win)
        df['result_val_for_player'] = df['result_for_player'].map({
            'win': 1.0,
            'agreed': 0.5,
            'timevsinsufficient': 0.5,
            'insufficient': 0.5,
            'stalemate': 0.5,
            'repetition': 0.5,
            'resigned': 0.0,
            'checkmated': 0.0,
            'timeout': 0.0,
            'abandoned': 0.0
        })
        
        # Calculate rating difference
        df['rating_difference'] = df['player_rating'] - df['opponent_rating']
        
        # Save advanced dataset
        df.to_csv(os.path.join('player_data', username, 'chess_dataset_adv.csv'), index=False)
        
    except Exception as e:
        raise Exception(f"Error creating advanced dataset: {str(e)}")

def check_cached_data(username: str) -> bool:
    """Check if required data files exist and are valid"""
    required_files = ["chess_dataset.csv", "chess_dataset_adv.csv"]
    player_dir = os.path.join('player_data', username)
    
    # Check if all required files exist
    if not all(os.path.exists(os.path.join(player_dir, file)) for file in required_files):
        return False
        
    # Check if files are not empty and are recent (less than 24 hours old)
    try:
        for file in required_files:
            file_path = os.path.join(player_dir, file)
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                return False
            # Check if file is recent
            if time.time() - os.path.getmtime(file_path) > 24 * 3600:  # 24 hours in seconds
                return False
        return True
    except Exception:
        return False

def driver_fn(username: str) -> None:
    """Main driver function to get and process chess data"""
    try:
        # Create user directory
        user_dir = os.path.join('player_data', username)
        os.makedirs(user_dir, exist_ok=True)
        
        # Check if we have valid cached data
        if check_cached_data(username):
            print(f"Using cached data for {username}")
            return
            
        # Get games data
        games = getGames(username)
        filterList(games, username)
        
        # Create datasets
        createDataset(games, username)
        createAdvancedDataset(username)  # Create advanced dataset for prediction
        
    except Exception as e:
        raise Exception(f"Error in driver function: {str(e)}")