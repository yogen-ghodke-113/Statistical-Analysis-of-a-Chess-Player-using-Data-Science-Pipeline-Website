import os
import warnings
import logging
import sys
from typing import Dict, List, Optional
import gc
import threading
import _thread
import io
import chess
import chess.pgn

# Set up logging with file output
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'visualization.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # Append mode
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*50)  # Separator for new runs
logger.info("Starting new visualization run")

# Suppress warnings
warnings.filterwarnings('ignore')

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go

class TimeoutException(Exception):
    pass

def timeout_handler():
    _thread.interrupt_main()

def time_limit(timeout):
    """Windows-compatible timeout context manager using threading"""
    timer = threading.Timer(timeout, timeout_handler)
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out!")
    finally:
        timer.cancel()

def check_dependencies() -> bool:
    """Check if all required dependencies are installed"""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        import plotly.io as pio
        import chess
        import chess.svg
        import chess.pgn
        from cairosvg import svg2png
        
        # Configure matplotlib for non-interactive backend
        plt.switch_backend('Agg')
        
        return True
    except ImportError as e:
        logger.error(f"Missing required dependency: {str(e)}")
        return False

def save_plotly_figure(fig, filepath: str, scale: int = 3) -> None:
    """Save plotly figure with error handling"""
    try:
        import plotly.io as pio
        logger.info(f"Saving figure to {filepath}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Configure kaleido for better performance
        pio.kaleido.scope.chromium_args = (
            '--no-sandbox',
            '--disable-gpu',
            '--disable-dev-shm-usage',
            '--single-process'
        )
        
        # Save with minimal configuration
        fig.write_image(
            filepath,
            format='png',
            engine='kaleido',
            scale=2,
            width=1200,
            height=800
        )
        logger.info("Successfully saved figure")
        
    except Exception as e:
        logger.error(f"Could not save figure: {str(e)}")
        # Fallback to HTML
        try:
            html_path = filepath.replace('.png', '.html')
            fig.write_html(html_path)
            logger.info(f"Saved as HTML fallback: {html_path}")
        except Exception as e2:
            logger.error(f"HTML fallback also failed: {str(e2)}")

def fight(df: pd.DataFrame, username: str) -> None:
    """Generate game length analysis visualization for lost games"""
    try:
        logger.info("Starting game length analysis...")
        
        # Get last 100 lost games
        lost_games = df[df['result_for_opponent'] == "win"].tail(100)
        
        # Create DataFrame for plotting
        moves_df = pd.DataFrame({
            'game_no': range(len(lost_games)),
            'moves': lost_games['moves'].values
        })
        
        # Create figure with matplotlib
        plt.figure(figsize=(12, 8))
        plt.style.use('seaborn')
        
        # Create line plot with markers
        plt.plot(moves_df['game_no'], moves_df['moves'], 
                marker='o', markersize=6, linewidth=2, 
                color='#2196F3', markerfacecolor='white',
                markeredgecolor='#2196F3', markeredgewidth=1.5)
        
        # Customize the plot
        plt.title("Moves in Last 100 Lost Games", size=16, pad=20)
        plt.xlabel("Game Number", size=12, labelpad=10)
        plt.ylabel("Number of Moves", size=12, labelpad=10)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join('player_data', username, "fight.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Game length analysis visualization complete")
        
    except Exception as e:
        logger.error(f"Error in game length analysis: {str(e)}")
        raise

def wh_countplot(df: pd.DataFrame, username: str) -> None:
    """Generate opening analysis visualization"""
    try:
        # White games analysis
        white_df = df[df["played_as"] == "white"]
        white_op_freq = white_df['opening'].value_counts().head(20)
        
        # Set figure size and style
        plt.figure(figsize=(20, 15))
        sns.set(rc={'figure.figsize': (20, 15)})
        sns.set_style("darkgrid", {'axes.grid': False})
        
        # Create plot
        ax = sns.countplot(
            y='opening',
            data=white_df[white_df['opening'].isin(white_op_freq.index)],
            order=white_op_freq.index
        )
        
        # Style the plot
        ax.set_ylabel("")
        ax.set_xlabel("Frequency", size=20, labelpad=30)
        ax.tick_params(labelsize=17)
        
        # Save figure
        output_path = os.path.join('player_data', username, "top_op_wh.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in opening analysis: {str(e)}")
        raise

def bl_countplot(df: pd.DataFrame, username: str) -> None:
    """Generate opening analysis visualization for black pieces"""
    try:
        # Black games analysis
        black_df = df[df["played_as"] == "black"]
        black_op_freq = black_df['opening'].value_counts().head(20)
        
        # Set figure size and style
        plt.figure(figsize=(20, 15))
        sns.set(rc={'figure.figsize': (20, 15)})
        sns.set_style("darkgrid", {'axes.grid': False})
        
        # Create plot
        ax = sns.countplot(
            y='opening',
            data=black_df[black_df['opening'].isin(black_op_freq.index)],
            order=black_op_freq.index
        )
        
        # Style the plot
        ax.set_ylabel("")
        ax.set_xlabel("Frequency", size=20, labelpad=30)
        ax.tick_params(labelsize=17)
        
        # Save figure
        output_path = os.path.join('player_data', username, "top_op_bl.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in black opening analysis: {str(e)}")
        raise

def most_used_wh(df: pd.DataFrame, username: str) -> None:
    """Generate chess board visualizations for top 3 first moves as white"""
    try:
        logger.info("Creating top 3 first moves visualization...")
        import chess
        import chess.svg
        import chess.pgn
        from cairosvg import svg2png

        # Get top 3 moves as white
        white_df = df[df["played_as"] == "white"]
        top_move = white_df["first_move"].value_counts().to_frame()
        top_move_list = []
        ret_di = {}
        
        for move, row in top_move.head(3).iterrows():
            top_move_list.append(move)
            ret_di.update({move: row["first_move"]})
        
        # Generate board visualizations
        num = 0
        for x in top_move_list:
            num += 1
            board = chess.Board()
            board.push(chess.Move.from_uci(x))
            
            # Generate SVG and convert to PNG
            svg_code = chess.svg.board(board=board)
            output_path = os.path.join('player_data', username, f'top_opening_move_as_white_{num}.png')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            svg2png(bytestring=svg_code, write_to=output_path, scale=4.0)
        
        logger.info("Top 3 first moves visualization complete")
        
    except Exception as e:
        logger.error(f"Error in top 3 first moves visualization: {str(e)}")
        raise

def create_rating_ladder(df: pd.DataFrame, username: str) -> None:
    """Create rating progress visualization"""
    try:
        logger.info("Creating rating ladder visualization...")
        
        # Get last 150 blitz games
        blitz_games = df[df['rated'] & df['time_class'].isin(["blitz"])].tail(150)
        
        # Create figure with seaborn style
        plt.figure(figsize=(12, 6))
        sns.set_style("darkgrid", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
        
        # Create line plot
        ax = sns.lineplot(data=blitz_games, x=range(len(blitz_games)), y='player_rating', 
                         color='#FF4D4D', linewidth=1, marker='o', markersize=4)
        
        # Customize the plot
        plt.title("Player's Elo Rating in the last 150 Rated Games", size=14, pad=10)
        plt.xlabel('game_no', size=12)
        plt.ylabel('player_rating', size=12)
        
        # Set background color
        ax.set_facecolor('#F0F2F6')
        plt.grid(True, alpha=0.3)
        
        # Adjust y-axis to show more detail in rating changes
        y_min = blitz_games['player_rating'].min() - 20
        y_max = blitz_games['player_rating'].max() + 20
        plt.ylim(y_min, y_max)
        
        # Add padding to the plot
        plt.margins(x=0.02)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join('player_data', username, "rating_ladder_red.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("Rating ladder visualization complete")
        
    except Exception as e:
        logger.error(f"Error in rating ladder: {str(e)}")
        raise

def create_result_distribution(df: pd.DataFrame, username: str) -> None:
    """Create pie chart of game results"""
    try:
        logger.info("Creating result distribution visualization...")
        
        # Count results
        result_counts = df['result_for_player'].value_counts()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.style.use('seaborn')
        
        # Create pie chart with clean styling
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(result_counts)))
        plt.pie(result_counts.values, labels=result_counts.index, 
               autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 12},
               colors=colors)
        
        # Remove unnecessary styling
        plt.axis('equal')
        
        # Save figure
        output_path = os.path.join('player_data', username, "result_pi.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        logger.info("Result distribution visualization complete")
        
    except Exception as e:
        logger.error(f"Error in result distribution: {str(e)}")
        raise

def create_time_control_dist(df: pd.DataFrame, username: str) -> None:
    """Create time control distribution visualization"""
    try:
        logger.info("Creating time control distribution...")
        
        # Get time control counts
        time_counts = df['time_class'].value_counts()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Define colors for each time control
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC']
        
        # Create pie chart with clean styling
        patches, texts, autotexts = plt.pie(
            time_counts.values, 
            labels=time_counts.index,
            colors=colors[:len(time_counts)],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        # Add title
        plt.title('Time Control Distribution', size=16, pad=20)
        
        # Add legend
        plt.legend(
            patches,
            time_counts.index,
            title="Time Controls",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        # Equal aspect ratio ensures circular pie
        plt.axis('equal')
        
        # Save figure with extra space for legend
        output_path = os.path.join('player_data', username, "time_class.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("Time control visualization complete")
        
    except Exception as e:
        logger.error(f"Error in time control distribution: {str(e)}")
        raise

def create_color_results(df: pd.DataFrame, username: str) -> None:
    """Create results by color visualizations"""
    try:
        logger.info("Creating color results visualizations...")
        
        # Function to create donut chart
        def create_donut_chart(data, title, output_path):
            plt.figure(figsize=(10, 8))
            plt.style.use('seaborn')
            
            # Calculate percentages
            total = sum(data)
            percentages = [count/total * 100 for count in data]
            
            # Create pie chart
            plt.pie(percentages, 
                   labels=['win', 'draw', 'loss'],
                   colors=['#2ecc71', '#95a5a6', '#e74c3c'],  # Green, Gray, Red
                   autopct='%1.1f%%',
                   startangle=90,
                   textprops={'fontsize': 12})
            
            # Add a circle at the center to create donut effect
            centre_circle = plt.Circle((0, 0), 0.70, color='gray', fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            
            plt.title(title, size=14, pad=20)
            plt.axis('equal')
            
            # Save figure
            plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
        
        # White games
        white_games = df[df['played_as'] == 'white']
        white_wins = len(white_games[white_games['result_for_player'] == 'win'])
        white_draws = len(white_games[white_games['result_for_player'].isin(
            ['agreed', 'timevsinsufficient', 'insufficient', 'stalemate', 'repetition'])])
        white_losses = len(white_games[white_games['result_for_player'].isin(
            ['resigned', 'checkmated', 'timeout', 'abandoned'])])
        
        # Create white results donut chart
        output_path = os.path.join('player_data', username, "result_as_wh.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        create_donut_chart([white_wins, white_draws, white_losses], 
                        'Results as White', output_path)
        
        # Black games
        black_games = df[df['played_as'] == 'black']
        black_wins = len(black_games[black_games['result_for_player'] == 'win'])
        black_draws = len(black_games[black_games['result_for_player'].isin(
            ['agreed', 'timevsinsufficient', 'insufficient', 'stalemate', 'repetition'])])
        black_losses = len(black_games[black_games['result_for_player'].isin(
            ['resigned', 'checkmated', 'timeout', 'abandoned'])])
        
        # Create black results donut chart
        output_path = os.path.join('player_data', username, "result_as_bl.png")
        create_donut_chart([black_wins, black_draws, black_losses], 
                        'Results as Black', output_path)
        
        logger.info("Color results visualizations complete")
        
    except Exception as e:
        logger.error(f"Error in color results: {str(e)}")
        raise

def create_top_5_openings(df: pd.DataFrame, username: str) -> None:
    """Create top 5 openings analysis for both colors"""
    try:
        logger.info("Creating top 5 openings analysis...")
        
        def wrap_labels(text, width=30):
            """Wrap text at specified width"""
            import textwrap
            return textwrap.fill(text, width=width)
        
        # White openings analysis
        white_df = df[df['played_as'] == 'white']
        white_op_freq = pd.DataFrame(white_df['opening'].value_counts())
        w_li = []
        for opening, row in white_op_freq.head(20).iterrows():
            w_li.append(opening)
            
        wh_op_name = []
        wh_win = []
        wh_draw = []
        wh_loss = []
        
        # Get top 5 openings data
        for x in range(5):
            wh_op_name.append(w_li[x])
            k = white_df[white_df["opening"] == w_li[x]]
            wh_win.append(k[k["result_for_player"].isin(["win"])].shape[0])
            wh_draw.append(k[k["result_for_player"].isin(
                ["agreed", "timevsinsufficient", "insufficient", "stalemate", "repetition"])].shape[0])
            wh_loss.append(k[k["result_for_player"].isin(
                ["resigned", "checkmated", "timeout", "abandoned"])].shape[0])
        
        di2 = {
            'wins': wh_win,
            'draws': wh_draw,
            'losses': wh_loss
        }
        
        # Wrap opening names
        wrapped_names = [wrap_labels(name) for name in wh_op_name]
        most_used_openings = pd.DataFrame(di2, index=wrapped_names)
        
        plt.figure(figsize=(15, 8))  # Increased figure size
        ax = most_used_openings[["wins", "draws", "losses"]].plot.barh(
            rot=0, color=["green", "blue", "red"], stacked=False)
        ax.set_facecolor('xkcd:white')
        
        # Adjust plot spacing to accommodate wrapped labels
        plt.subplots_adjust(left=0.3)  # Increase space for labels
        
        # Format legend and labels
        ax.legend(prop={'size': 12}, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("Result of Top 5 Openings as White:", size=14, y=1.05)
        plt.xlabel("Number of Games", size=12, labelpad=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Add grid lines
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save white openings figure
        output_path = os.path.join('player_data', username, "result_top_5_wh.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Black openings analysis
        black_df = df[df['played_as'] == 'black']
        black_op_freq = pd.DataFrame(black_df['opening'].value_counts())
        b_li = []
        for opening, row in black_op_freq.head(20).iterrows():
            b_li.append(opening)
            
        bl_op_name = []
        bl_win = []
        bl_draw = []
        bl_loss = []
        
        # Get top 5 openings data
        for x in range(5):
            bl_op_name.append(b_li[x])
            k = black_df[black_df["opening"] == b_li[x]]
            bl_win.append(k[k["result_for_player"].isin(["win"])].shape[0])
            bl_draw.append(k[k["result_for_player"].isin(
                ["agreed", "timevsinsufficient", "insufficient", "stalemate", "repetition"])].shape[0])
            bl_loss.append(k[k["result_for_player"].isin(
                ["resigned", "checkmated", "timeout", "abandoned"])].shape[0])
        
        di2 = {
            'wins': bl_win,
            'draws': bl_draw,
            'losses': bl_loss
        }
        
        # Wrap opening names
        wrapped_names = [wrap_labels(name) for name in bl_op_name]
        most_used_openings = pd.DataFrame(di2, index=wrapped_names)
        
        plt.figure(figsize=(15, 8))  # Increased figure size
        ax = most_used_openings[["wins", "draws", "losses"]].plot.barh(
            rot=0, color=["green", "blue", "red"], stacked=False)
        ax.set_facecolor('xkcd:white')
        
        # Adjust plot spacing to accommodate wrapped labels
        plt.subplots_adjust(left=0.3)  # Increase space for labels
        
        # Format legend and labels
        ax.legend(prop={'size': 12}, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("Result of Top 5 Openings as Black:", size=14, y=1.05)
        plt.xlabel("Number of Games", size=12, labelpad=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Add grid lines
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save black openings figure
        output_path = os.path.join('player_data', username, "result_top_5_bl.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Top 5 openings analysis complete")
        
    except Exception as e:
        logger.error(f"Error in top 5 openings analysis: {str(e)}")
        raise

def create_overall_results(df: pd.DataFrame, username: str) -> None:
    """Create overall results visualization showing detailed breakdown of game results"""
    try:
        logger.info("Creating overall results visualization...")
        
        # Get all possible result types and their counts
        all_results = ['win', 'resigned', 'checkmated', 'timeout', 'repetition', 
                      'abandoned', 'stalemate', 'time vs\ninsufficient', 'insufficient', 'agreed']
        
        # Map the original values to display values
        result_map = {
            'timevsinsufficient': 'time vs\ninsufficient',
            'win': 'win',
            'resigned': 'resigned',
            'checkmated': 'checkmated',
            'timeout': 'timeout',
            'repetition': 'repetition',
            'abandoned': 'abandoned',
            'stalemate': 'stalemate',
            'insufficient': 'insufficient',
            'agreed': 'agreed'
        }
        
        # Create a copy of the result column with mapped values
        df['result_display'] = df['result_for_player'].map(result_map)
        result_counts = df['result_display'].value_counts()
        
        # Create a Series with all results, filling missing values with 0
        most_frequently_opening = pd.Series(0, index=all_results)
        most_frequently_opening.update(result_counts)
        
        plt.figure(figsize=(20, 10))  # Increased figure size
        
        # Create custom color palette from dark bluish-gray to lighter blues
        colors = sns.color_palette("Blues_d", n_colors=len(all_results))
        colors.reverse()  # Reverse to match the original dark-to-light pattern
        
        # Create bar plot with custom styling
        opening = sns.barplot(x=most_frequently_opening.index, 
                            y=most_frequently_opening.values,
                            palette=colors)
        
        # Style the plot
        plt.title("Overall Game Results Distribution", fontsize=24, pad=20)  # Added title
        plt.ylabel("Number of Games", fontsize=20, labelpad=15, weight='bold')
        plt.xlabel("Game Result", fontsize=20, labelpad=15, weight='bold')
        
        # Add value labels on top of bars with increased size
        for p in opening.patches:
            height = p.get_height()
            text = str(int(height))
            opening.text(p.get_x() + p.get_width() / 2, height + 1, 
                       text, ha="center", fontsize=14, fontweight='bold')
        
        # Set background color and remove grid
        opening.set_facecolor('xkcd:white')
        plt.grid(False)
        
        # Set y-axis to start from 0 with more padding for labels
        plt.ylim(0, max(most_frequently_opening.values) * 1.15)
        
        # Increase tick label size and make horizontal
        plt.xticks(fontsize=16, rotation=0, ha='center')
        plt.yticks(fontsize=16)
        
        # Adjust layout to prevent label cutoff
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
        
        # Save the figure with higher DPI
        output_path = os.path.join('player_data', username, "overall_results.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("Overall results visualization complete")
        
    except Exception as e:
        logger.error(f"Error in overall results: {str(e)}")
        raise

def wh_heatmap_beg(df: pd.DataFrame, username: str) -> None:
    """Create heatmap for starting squares as white"""
    di = {
        "a1": [0, 0, 0, 0, 0, 0, 0, 0],
        "b1": [0, 0, 0, 0, 0, 0, 0, 0],
        "c1": [0, 0, 0, 0, 0, 0, 0, 0],
        "d1": [0, 0, 0, 0, 0, 0, 0, 0],
        "e1": [0, 0, 0, 0, 0, 0, 0, 0],
        "f1": [0, 0, 0, 0, 0, 0, 0, 0],
        "g1": [0, 0, 0, 0, 0, 0, 0, 0],
        "h1": [0, 0, 0, 0, 0, 0, 0, 0],
        "a2": [0, 0, 0, 0, 0, 0, 0, 0],
        "b2": [0, 0, 0, 0, 0, 0, 0, 0],
        "c2": [0, 0, 0, 0, 0, 0, 0, 0],
        "d2": [0, 0, 0, 0, 0, 0, 0, 0],
        "e2": [0, 0, 0, 0, 0, 0, 0, 0],
        "f2": [0, 0, 0, 0, 0, 0, 0, 0],
        "g2": [0, 0, 0, 0, 0, 0, 0, 0],
        "h2": [0, 0, 0, 0, 0, 0, 0, 0],
    }

    for index, row in df[df['played_as'] == "white"].iterrows():
        di[row["first_move"][0] + "1"][int(row["first_move"][1]) - 1] += 1
        di[row["first_move"][2] + "2"][int(row["first_move"][3]) - 1] += 1

    row = ['1', '2', '3', '4', '5', '6', '7', '8']
    a = di["a1"]
    b = di["b1"]
    c = di["c1"]
    d = di["d1"]
    e = di["e1"]
    f = di["f1"]
    g = di["g1"]
    h = di["h1"]

    mlist = [row, a, b, c, d, e, f, g, h]
    for lists in mlist:
        lists.reverse()

    board_open = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d,
                             'e': e, 'f': f, 'g': g, 'h': h}, index=row)

    plt.figure(figsize=(10, 10))
    board = sns.heatmap(board_open, cmap='Reds', square=True, linewidths=.1, linecolor='black')
    board.set_title('Starting Square Heatmap as White', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_starting_white.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def wh_heatmap_end(df: pd.DataFrame, username: str) -> None:
    """Create heatmap for landing squares as white"""
    di = {
        "a1": [0, 0, 0, 0, 0, 0, 0, 0],
        "b1": [0, 0, 0, 0, 0, 0, 0, 0],
        "c1": [0, 0, 0, 0, 0, 0, 0, 0],
        "d1": [0, 0, 0, 0, 0, 0, 0, 0],
        "e1": [0, 0, 0, 0, 0, 0, 0, 0],
        "f1": [0, 0, 0, 0, 0, 0, 0, 0],
        "g1": [0, 0, 0, 0, 0, 0, 0, 0],
        "h1": [0, 0, 0, 0, 0, 0, 0, 0],
        "a2": [0, 0, 0, 0, 0, 0, 0, 0],
        "b2": [0, 0, 0, 0, 0, 0, 0, 0],
        "c2": [0, 0, 0, 0, 0, 0, 0, 0],
        "d2": [0, 0, 0, 0, 0, 0, 0, 0],
        "e2": [0, 0, 0, 0, 0, 0, 0, 0],
        "f2": [0, 0, 0, 0, 0, 0, 0, 0],
        "g2": [0, 0, 0, 0, 0, 0, 0, 0],
        "h2": [0, 0, 0, 0, 0, 0, 0, 0],
    }

    for index, row in df[df['played_as'] == "white"].iterrows():
        di[row["first_move"][2] + "1"][int(row["first_move"][3]) - 1] += 1
        di[row["first_move"][2] + "2"][int(row["first_move"][3]) - 1] += 1

    row = ['1', '2', '3', '4', '5', '6', '7', '8']
    a = di["a2"]
    b = di["b2"]
    c = di["c2"]
    d = di["d2"]
    e = di["e2"]
    f = di["f2"]
    g = di["g2"]
    h = di["h2"]

    mlist = [row, a, b, c, d, e, f, g, h]
    for lists in mlist:
        lists.reverse()

    board_open = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d,
                             'e': e, 'f': f, 'g': g, 'h': h}, index=row)

    plt.figure(figsize=(10, 10))
    board = sns.heatmap(board_open, cmap='Reds', square=True, linewidths=.1, linecolor='black')
    board.set_title('Landing Square Heatmap as White', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_landing_white.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def bl_heatmap_beg(df: pd.DataFrame, username: str) -> None:
    """Create heatmap for starting squares as black"""
    # Initialize the board matrix directly (8x8)
    board_matrix = np.zeros((8, 8))

    # Process only games where user played as black
    black_games = df[df['played_as'] == "black"]
    
    for _, row in black_games.iterrows():
        pgn = chess.pgn.read_game(io.StringIO(row['PGN']))
        if pgn:
            moves = list(pgn.mainline_moves())
            if len(moves) >= 2:  # Make sure there are at least 2 moves (white's first move and black's response)
                black_move = moves[1]  # Get black's first move (second move in the game)
                from_square = chess.square_name(black_move.from_square)
                
                # Get file and rank indices (0-7)
                file_idx = ord(from_square[0]) - ord('a')  # Convert a-h to 0-7
                rank_idx = 8 - int(from_square[1])  # Convert 1-8 to 7-0 (flipped for display)
                
                # Increment the count in the matrix
                board_matrix[rank_idx][file_idx] += 1

    # Create DataFrame for seaborn
    board_df = pd.DataFrame(
        board_matrix,
        index=['8', '7', '6', '5', '4', '3', '2', '1'],  # Ranks from top to bottom
        columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']  # Files from left to right
    )

    plt.figure(figsize=(10, 10))
    board = sns.heatmap(board_df, cmap='Blues', square=True, linewidths=.1, linecolor='black')
    board.set_title('Starting Square Heatmap as Black', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_starting_black.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def bl_heatmap_end(df: pd.DataFrame, username: str) -> None:
    """Create heatmap for landing squares as black"""
    # Initialize the board matrix directly (8x8)
    board_matrix = np.zeros((8, 8))

    # Process only games where user played as black
    black_games = df[df['played_as'] == "black"]
    
    for _, row in black_games.iterrows():
        pgn = chess.pgn.read_game(io.StringIO(row['PGN']))
        if pgn:
            moves = list(pgn.mainline_moves())
            if len(moves) >= 2:  # Make sure there are at least 2 moves (white's first move and black's response)
                black_move = moves[1]  # Get black's first move (second move in the game)
                to_square = chess.square_name(black_move.to_square)
                
                # Get file and rank indices (0-7)
                file_idx = ord(to_square[0]) - ord('a')  # Convert a-h to 0-7
                rank_idx = 8 - int(to_square[1])  # Convert 1-8 to 7-0 (flipped for display)
                
                # Increment the count in the matrix
                board_matrix[rank_idx][file_idx] += 1

    # Create DataFrame for seaborn
    board_df = pd.DataFrame(
        board_matrix,
        index=['8', '7', '6', '5', '4', '3', '2', '1'],  # Ranks from top to bottom
        columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']  # Files from left to right
    )

    plt.figure(figsize=(10, 10))
    board = sns.heatmap(board_df, cmap='Blues', square=True, linewidths=.1, linecolor='black')
    board.set_title('Landing Square Heatmap as Black', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_landing_black.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_combined_heatmaps(df: pd.DataFrame, username: str) -> None:
    """Create and combine all heatmaps"""
    try:
        logger.info("Creating combined heatmap visualizations...")
        
        # Create individual heatmaps
        wh_heatmap_beg(df, username)
        wh_heatmap_end(df, username)
        bl_heatmap_beg(df, username)
        bl_heatmap_end(df, username)
        
        # Create combined figure for white heatmaps
        plt.figure(figsize=(20, 10))
        
        # White starting squares
        plt.subplot(1, 2, 1)
        img1 = plt.imread(os.path.join('player_data', username, "heatmap_starting_white.png"))
        plt.imshow(img1)
        plt.axis('off')
        
        # White landing squares
        plt.subplot(1, 2, 2)
        img2 = plt.imread(os.path.join('player_data', username, "heatmap_landing_white.png"))
        plt.imshow(img2)
        plt.axis('off')
        
        plt.suptitle("Starting and Landing Square Heatmaps as White", size=24, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join('player_data', username, "heatmap_combined_white.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create combined figure for black heatmaps
        plt.figure(figsize=(20, 10))
        
        # Black starting squares
        plt.subplot(1, 2, 1)
        img3 = plt.imread(os.path.join('player_data', username, "heatmap_starting_black.png"))
        plt.imshow(img3)
        plt.axis('off')
        
        # Black landing squares
        plt.subplot(1, 2, 2)
        img4 = plt.imread(os.path.join('player_data', username, "heatmap_landing_black.png"))
        plt.imshow(img4)
        plt.axis('off')
        
        plt.suptitle("Starting and Landing Square Heatmaps as Black", size=24, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join('player_data', username, "heatmap_combined_black.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info("Combined heatmap visualizations complete")
        
    except Exception as e:
        logger.error(f"Error in combined heatmaps: {str(e)}")
        raise

def driver_fn(username: str) -> None:
    """Main driver function for visualizations"""
    try:
        logger.info(f"Starting game analysis for user: {username}")
        
        # Configure matplotlib for non-interactive backend
        plt.switch_backend('Agg')
        
        if not check_dependencies():
            raise Exception("Missing required dependencies")
            
        # Ensure directory exists and is absolute
        user_dir = os.path.join('player_data', username)
        os.makedirs(user_dir, exist_ok=True)
        
        logger.info(f"Loading data for {username}")
        df_path = os.path.join(user_dir, 'chess_dataset.csv')
        
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Dataset not found at {df_path}. Please ensure the data has been downloaded first.")
            
        df = pd.read_csv(df_path)
        if len(df) == 0:
            raise ValueError("Dataset is empty")
            
        logger.info(f"Loaded {len(df)} games")
        
        # Add derived columns
        logger.info("Processing data...")
        df["rating_difference"] = df["player_rating"] - df["opponent_rating"]
        df["moves"] = pd.to_numeric(df["moves"], errors='coerce')
        
        # Generate all visualizations
        visualization_functions = [
            (fight, "fight analysis"),
            (wh_countplot, "white opening analysis"),
            (bl_countplot, "black opening analysis"),
            (most_used_wh, "top 3 first moves"),
            (create_rating_ladder, "rating progress"),
            (create_time_control_dist, "time control distribution"),
            (create_color_results, "color results"),
            (create_top_5_openings, "top 5 openings analysis"),
            (create_overall_results, "overall results"),
            (wh_heatmap_beg, "starting squares as white"),
            (wh_heatmap_end, "landing squares as white"),
            (bl_heatmap_beg, "starting squares as black"),
            (bl_heatmap_end, "landing squares as black"),
            (create_combined_heatmaps, "combined heatmaps")
        ]
        
        successful_visualizations = []
        failed_visualizations = []
        
        for viz_func, description in visualization_functions:
            try:
                logger.info(f"Starting {description}...")
                
                # Clear memory before each visualization
                plt.close('all')
                gc.collect()
                
                viz_func(df, username)
                logger.info(f"Completed {description}")
                successful_visualizations.append(description)
                
            except Exception as e:
                logger.error(f"Error in {description}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                failed_visualizations.append(f"{description} ({type(e).__name__})")
                continue
                
        # Log summary
        logger.info("\nVisualization Summary:")
        logger.info(f"Successfully completed: {len(successful_visualizations)}")
        for viz in successful_visualizations:
            logger.info(f"✓ {viz}")
            
        if failed_visualizations:
            logger.info(f"\nFailed visualizations: {len(failed_visualizations)}")
            for viz in failed_visualizations:
                logger.info(f"✗ {viz}")
                
    except Exception as e:
        logger.error(f"Fatal error in visualization process: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise  # Re-raise to ensure the error is propagated

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Please provide a username as argument")
        sys.exit(1)
    try:
        driver_fn(sys.argv[1])
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
