import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List

def heatmap(df: pd.DataFrame, username: str) -> None:
    """Create correlation heatmap for numeric columns only"""
    try:
        # Select only numeric columns for correlation
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        correlation_df = df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(20, 15))
        sns.set(font_scale=1.5)
        k = sns.heatmap(correlation_df, annot=True, square=False)
        k.get_figure().savefig(os.path.join(username, "corr_heatmap.png"), bbox_inches='tight', dpi=300)
        plt.close()  # Close the figure to free memory
        
    except Exception as e:
        print(f"Error creating heatmap: {str(e)}")

def driver_fn(username: str) -> None:
    """Process data and create heatmap"""
    try:
        # Read the dataset
        df = pd.read_csv(os.path.join(username, 'chess_dataset.csv'))
        
        # Add rating difference
        df["rating_difference"] = df["player_rating"] - df["opponent_rating"]
        
        # Convert game results to numeric values
        result_map = {
            "win": 1.0,
            "agreed": 0.5, 
            "timevsinsufficient": 0.5,
            "insufficient": 0.5,
            "stalemate": 0.5,
            "repetition": 0.5,
            "resigned": 0.0,
            "checkmated": 0.0,
            "timeout": 0.0,
            "abandoned": 0.0
        }
        
        df['result_val_for_player'] = df['result_for_player'].map(result_map)
        
        # Create heatmap
        heatmap(df, username)
        
    except Exception as e:
        print(f"Error in driver function: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        driver_fn(sys.argv[1])
    else:
        print("Please provide a username as argument")