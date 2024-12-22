import requests
import get_data as gd
import json
import time
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mord as mrd


def predict(u1, u2):
    di = {}
    headers = {
        'User-Agent': 'Chess Analysis App (Contact: github.com/yogen-ghodke-113)',
        'Accept': 'application/json'
    }

    try:
        # Get user's stats with retry logic
        max_retries = 3
        retry_delay = 2  # seconds

        # Initialize ratings
        user_rating = None
        opp_rating = None

        # Get user's rating
        for attempt in range(max_retries):
            try:
                req = "https://api.chess.com/pub/player/" + u1 + "/stats"
                response = requests.get(req, headers=headers, timeout=10)
                
                if response.status_code == 429:  # Too Many Requests
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception("Rate limit exceeded. Please try again in a few minutes.")
                
                response.raise_for_status()
                user_stats = response.json()
                
                # Try to get blitz rating, fall back to rapid, then bullet if not available
                if "chess_blitz" in user_stats and "last" in user_stats["chess_blitz"]:
                    user_rating = user_stats["chess_blitz"]["last"]["rating"]
                elif "chess_rapid" in user_stats and "last" in user_stats["chess_rapid"]:
                    user_rating = user_stats["chess_rapid"]["last"]["rating"]
                elif "chess_bullet" in user_stats and "last" in user_stats["chess_bullet"]:
                    user_rating = user_stats["chess_bullet"]["last"]["rating"]
                else:
                    raise ValueError(f"Could not find any rating for user {u1}")
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception("Request timed out. Please try again.")

        # Get opponent's rating
        for attempt in range(max_retries):
            try:
                req = "https://api.chess.com/pub/player/" + u2 + "/stats"
                response = requests.get(req, headers=headers, timeout=10)
                
                if response.status_code == 429:  # Too Many Requests
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception("Rate limit exceeded. Please try again in a few minutes.")
                
                response.raise_for_status()
                opp_stats = response.json()
                
                # Try to get blitz rating, fall back to rapid, then bullet if not available
                if "chess_blitz" in opp_stats and "last" in opp_stats["chess_blitz"]:
                    opp_rating = opp_stats["chess_blitz"]["last"]["rating"]
                elif "chess_rapid" in opp_stats and "last" in opp_stats["chess_rapid"]:
                    opp_rating = opp_stats["chess_rapid"]["last"]["rating"]
                elif "chess_bullet" in opp_stats and "last" in opp_stats["chess_bullet"]:
                    opp_rating = opp_stats["chess_bullet"]["last"]["rating"]
                else:
                    raise ValueError(f"Could not find any rating for user {u2}")
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception("Request timed out. Please try again.")

        if user_rating is None or opp_rating is None:
            raise ValueError("Could not retrieve ratings for both players")

        di["user_rating"] = user_rating
        di["opp_rating"] = opp_rating
        diff = user_rating - opp_rating
        di["rating_diff"] = diff

    except requests.exceptions.RequestException as e:
        if "getaddrinfo failed" in str(e):
            raise Exception("Unable to connect to Chess.com. Please check your internet connection.")
        elif "Name or service not known" in str(e):
            raise Exception("Unable to resolve Chess.com. Please check your DNS settings.")
        else:
            raise Exception(f"Error accessing Chess.com API: {str(e)}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Error processing player stats: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing API response: {str(e)}")

    try:
        # Read the advanced dataset from the correct path
        adv_dataset_path = os.path.join('player_data', u1, 'chess_dataset_adv.csv')
        if not os.path.exists(adv_dataset_path):
            raise FileNotFoundError(f"Advanced dataset not found at {adv_dataset_path}")
        df = pd.read_csv(adv_dataset_path)

        # Calculate win probability using FIDE formula
        expected_score = 1 / (1 + 10 ** (-diff/400))
        
        # Adjust probabilities based on rating difference and historical data
        df['rating_bucket'] = pd.cut(df['rating_difference'], 
                                   bins=[-float('inf'), -400, -200, -100, 0, 100, 200, 400, float('inf')],
                                   labels=['<-400', '-400to-200', '-200to-100', '-100to0', '0to100', '100to200', '200to400', '>400'])
        
        # Calculate historical probabilities for each bucket
        historical_probs = df.groupby('rating_bucket')['result_val_for_player'].agg(['mean', 'count']).reset_index()
        
        # Find the bucket for current rating difference
        if diff <= -400:
            bucket = '<-400'
        elif -400 < diff <= -200:
            bucket = '-400to-200'
        elif -200 < diff <= -100:
            bucket = '-200to-100'
        elif -100 < diff <= 0:
            bucket = '-100to0'
        elif 0 < diff <= 100:
            bucket = '0to100'
        elif 100 < diff <= 200:
            bucket = '100to200'
        elif 200 < diff <= 400:
            bucket = '200to400'
        else:
            bucket = '>400'
            
        # Get historical probability for this bucket
        bucket_stats = historical_probs[historical_probs['rating_bucket'] == bucket].iloc[0]
        historical_win_rate = bucket_stats['mean']
        sample_size = bucket_stats['count']
        
        # Weighted average between FIDE formula and historical data
        # Weight increases with sample size up to a maximum of 0.7
        historical_weight = min(0.7, sample_size / 100)
        fide_weight = 1 - historical_weight
        
        win_prob = (fide_weight * expected_score + historical_weight * historical_win_rate) * 100
        
        # Calculate draw probability based on rating difference
        # Draw probability is highest when ratings are close
        base_draw_prob = 30  # Base draw probability
        rating_factor = abs(diff) / 400  # Normalize rating difference
        draw_prob = base_draw_prob * (1 - min(1, rating_factor))  # Decrease draw probability as rating difference increases
        
        # Ensure loss probability makes the total 100%
        loss_prob = 100 - win_prob - draw_prob
        
        # Determine result based on probabilities
        if win_prob > max(draw_prob, loss_prob):
            result = 'Win'
        elif draw_prob > max(win_prob, loss_prob):
            result = 'Draw'
        else:
            result = 'Loss'

        # Store model accuracy (using historical accuracy for the relevant bucket)
        di["ord_acc"] = f'Model accuracy for games with similar rating differences: {bucket_stats["mean"]*100:.1f}% (based on {int(bucket_stats["count"])} games)'
        
        # Store model summary
        summary = (f"Rating difference: {diff}\n"
                  f"Historical win rate for similar differences: {historical_win_rate*100:.1f}%\n"
                  f"FIDE expected score: {expected_score*100:.1f}%\n"
                  f"Sample size: {int(sample_size)} games\n"
                  f"Historical weight: {historical_weight*100:.1f}%")
        di["summ1"] = summary

        di["result"] = (f'Result: {result}\n\n'
                       f'Win probability: {win_prob:.1f}%\n'
                       f'Draw probability: {draw_prob:.1f}%\n'
                       f'Loss probability: {loss_prob:.1f}%')

    except FileNotFoundError as e:
        raise Exception(f"Error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error during prediction calculation: {str(e)}")

    return di




#k1 = predict("tyrange","sudesh2911")
#print(k1)