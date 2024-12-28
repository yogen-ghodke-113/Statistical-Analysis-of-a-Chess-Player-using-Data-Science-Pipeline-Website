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
import cairosvg

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
        
        # Add frequency counts at the end of each bar
        for i, v in enumerate(white_op_freq.values):
            ax.text(v + 0.1, i, str(v), va='center', fontsize=12)
        
        # Style the plot
        ax.set_ylabel("")
        ax.set_xlabel("Frequency", size=20, labelpad=30)
        ax.tick_params(labelsize=17)
        
        # Extend x-axis to make room for labels
        plt.xlim(0, max(white_op_freq.values) * 1.1)
        
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
        
        # Add frequency counts at the end of each bar
        for i, v in enumerate(black_op_freq.values):
            ax.text(v + 0.1, i, str(v), va='center', fontsize=12)
        
        # Style the plot
        ax.set_ylabel("")
        ax.set_xlabel("Frequency", size=20, labelpad=30)
        ax.tick_params(labelsize=17)
        
        # Extend x-axis to make room for labels
        plt.xlim(0, max(black_op_freq.values) * 1.1)
        
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
        
        # Get top 3 moves as white
        white_df = df[df["played_as"] == "white"]
        if white_df.empty:
            logger.warning("No games found as white pieces")
            return
            
        # Extract and count first moves
        first_moves = white_df["first_move"].value_counts()
        logger.info(f"Found first moves: {first_moves.to_dict()}")
        
        if first_moves.empty:
            logger.warning("No first moves found in white games")
            return

        # Process top 3 moves
        for i, (move, count) in enumerate(first_moves.head(3).items(), 1):
            try:
                # Create a new board
                board = chess.Board()
                
                # Parse move using python-chess's built-in parser
                try:
                    chess_move = board.parse_san(move)
                    board.push(chess_move)
                except ValueError:
                    logger.warning(f"Could not parse move {move}")
                    continue
                    
                # Generate SVG with Lichess-style colors
                svg_content = chess.svg.board(
                    board=board,
                    size=400,
                    coordinates=True,
                    colors={
                        'square light': '#f0d9b5',  # Lichess brown light squares
                        'square dark': '#b58863',   # Lichess brown dark squares
                        'square light lastmove': '#cdd26a',  # Lichess last move highlight light
                        'square dark lastmove': '#aaa23a',   # Lichess last move highlight dark
                        'margin': 'none',
                        'coord': '#666666'          # Lichess coordinate color
                    }
                )
                
                # Save SVG first
                svg_path = os.path.join('player_data', username, f'top_opening_move_as_white_{i}.svg')
                with open(svg_path, 'w') as f:
                    f.write(svg_content)
                
                # Convert to PNG
                png_path = os.path.join('player_data', username, f'top_opening_move_as_white_{i}.png')
                cairosvg.svg2png(
                    url=svg_path,
                    write_to=png_path,
                    scale=2.0  # Increase quality
                )
                
                logger.info(f"Successfully created visualization for move {i}: {move}")
                
            except Exception as e:
                logger.error(f"Error creating visualization for move {i}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in top moves visualization: {str(e)}")
        raise

def most_used_bl(df: pd.DataFrame, username: str) -> None:
    """Generate chess board visualizations for top 3 first replies as black"""
    try:
        logger.info("Creating top 3 black replies visualization...")
        import chess
        import chess.svg
        import chess.pgn
        
        # Get games where user played as black
        black_games = df[df['played_as'] == "black"]
        if black_games.empty:
            logger.warning("No games found as black pieces")
            return
            
        # Dictionary to store black's first moves
        black_replies = {}
        
        # Process each game to extract black's first move
        for _, row in black_games.iterrows():
            pgn = chess.pgn.read_game(io.StringIO(row['PGN']))
            if pgn:
                moves = list(pgn.mainline_moves())
                if len(moves) >= 2:  # Make sure there are at least 2 moves
                    board = chess.Board()
                    board.push(moves[0])  # Apply white's first move
                    black_move = moves[1]  # Get black's response
                    move_san = board.san(black_move)  # Get move in SAN notation
                    black_replies[move_san] = black_replies.get(move_san, 0) + 1

        if not black_replies:
            logger.warning("No black replies found")
            return

        # Sort replies by frequency
        sorted_replies = sorted(black_replies.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Found black replies: {dict(sorted_replies)}")

        # Process top 3 replies
        for i, (move, count) in enumerate(sorted_replies[:3], 1):
            try:
                # Create a new board
                board = chess.Board()
                
                # Make a common first move for white (e4) to show black's reply context
                board.push_san("e4")  # We'll show black's replies against e4
                
                # Parse black's move
                try:
                    chess_move = board.parse_san(move)
                    board.push(chess_move)
                except ValueError:
                    logger.warning(f"Could not parse move {move}")
                    continue
                    
                # Generate SVG with Lichess-style colors
                svg_content = chess.svg.board(
                    board=board,
                    size=400,
                    coordinates=True,
                    orientation=chess.BLACK,  # Show board from black's perspective
                    colors={
                        'square light': '#f0d9b5',  # Lichess brown light squares
                        'square dark': '#b58863',   # Lichess brown dark squares
                        'square light lastmove': '#cdd26a',  # Lichess last move highlight light
                        'square dark lastmove': '#aaa23a',   # Lichess last move highlight dark
                        'margin': 'none',
                        'coord': '#666666'          # Lichess coordinate color
                    }
                )
                
                # Save SVG first
                svg_path = os.path.join('player_data', username, f'top_reply_move_as_black_{i}.svg')
                with open(svg_path, 'w') as f:
                    f.write(svg_content)
                
                # Convert to PNG
                png_path = os.path.join('player_data', username, f'top_reply_move_as_black_{i}.png')
                cairosvg.svg2png(
                    url=svg_path,
                    write_to=png_path,
                    scale=2.0  # Increase quality
                )
                
                logger.info(f"Successfully created visualization for black reply {i}: {move}")
                
            except Exception as e:
                logger.error(f"Error creating visualization for move {i}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in top black replies visualization: {str(e)}")
        raise

def create_rating_ladder(df: pd.DataFrame, username: str) -> None:
    """Create rating progress visualization showing last 150 games for each time control"""
    try:
        logger.info("Creating rating ladder visualization...")
        
        # Define time controls we want to track (excluding daily)
        time_controls = ["bullet", "blitz", "rapid"]
        
        # Create figure with seaborn style
        plt.figure(figsize=(12, 6))
        sns.set_style("darkgrid", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
        
        # Track if we have any data to plot
        has_data = False
        
        # Define markers and colors for each time control with better visibility
        style_map = {
            'bullet': ('X', '#FF3333'),  # Changed 'x' to 'X' for bigger marker, Bright red
            'blitz': ('o', '#0066CC'),   # Deep blue circle
            'rapid': ('s', '#00CC66'),   # Deep green square
        }
        
        # Process each time control
        for time_class in time_controls:
            # Get last 150 rated games for this time control
            games = df[
                (df['rated']) & 
                (df['time_class'] == time_class)
            ].tail(150)
            
            # Skip if no games for this time control
            if games.empty:
                logger.info(f"No rated {time_class} games found")
                continue
                
            # Remove any NaN values from player_rating
            games = games.dropna(subset=['player_rating'])
            
            if not games.empty:
                has_data = True
                marker, color = style_map[time_class]
                # Create line plot
                sns.lineplot(
                    data=games, 
                    x=range(len(games)), 
                    y='player_rating',
                    label=time_class.capitalize(),
                    marker=marker,
                    color=color,
                    markersize=8,  # Increased from 6
                    markeredgewidth=2,  # Added to make markers more visible
                    linewidth=1.5
                )
        
        if not has_data:
            logger.warning("No rated games found in any time control")
            return
            
        # Customize the plot
        plt.title("Rating Progress by Time Control (Last 150 Games per Type)", size=14, pad=20)
        plt.xlabel("Game Number", size=12)
        plt.ylabel("Rating", size=12)
        
        # Add legend with custom title
        plt.legend(
            title="Time Control",
            title_fontsize=12,
            fontsize=10,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join('player_data', username, "rating_ladder_red.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Rating ladder visualization created successfully")
        
    except Exception as e:
        logger.error(f"Error creating rating ladder: {str(e)}")
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
    """Create top 5 most successful and least successful openings analysis for both colors"""
    try:
        logger.info("Creating top 5 openings analysis...")
        
        # Ensure output directory exists
        output_dir = os.path.join('player_data', username)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        def wrap_labels(text, width=30):
            """Wrap text at specified width"""
            import textwrap
            return textwrap.fill(text, width=width)
        
        def calculate_opening_stats(color_df):
            """Calculate stats for openings using points-based system"""
            opening_stats = []
            logger.info(f"Processing {len(color_df)} games")
            
            # Get value counts of openings first
            opening_counts = color_df['opening'].value_counts()
            logger.info(f"Unique openings found: {len(opening_counts)}")
            
            # Process openings
            for opening in opening_counts.index:
                games = color_df[color_df['opening'] == opening]
                total_games = len(games)
                
                wins = len(games[games['result_for_player'] == 'win'])
                draws = len(games[games['result_for_player'].isin(
                    ['agreed', 'timevsinsufficient', 'insufficient', 'stalemate', 'repetition'])])
                losses = len(games[games['result_for_player'].isin(
                    ['resigned', 'checkmated', 'timeout', 'abandoned'])])
                
                # Calculate points using the weighted system
                points = (wins * 1.0) + (draws * 0.5) + (losses * 0)
                points_percentage = (points / total_games) * 100
                
                opening_stats.append({
                    'opening': opening,
                    'points_percentage': points_percentage,
                    'total_games': total_games,
                    'wins': wins,
                    'draws': draws,
                    'losses': losses,
                    'points': points
                })
            
            logger.info(f"Processed {len(opening_stats)} openings")
            return opening_stats
        
        def create_opening_chart(stats, title, output_path, best=True):
            """Create bar chart for openings"""
            try:
                logger.info(f"Creating chart: {title}")
                logger.info(f"Number of stats available: {len(stats)}")
                
                # Get top 5 or bottom 5 based on points percentage
                selected_stats = stats[:5] if best else stats[-5:]
                if not best:
                    selected_stats = selected_stats[::-1]  # Reverse order for worst openings
                
                logger.info(f"Selected {len(selected_stats)} stats for visualization")
                
                # Prepare data for plotting
                openings = [wrap_labels(s['opening'], width=25) for s in selected_stats]
                wins = [s['wins'] for s in selected_stats]
                draws = [s['draws'] for s in selected_stats]
                losses = [s['losses'] for s in selected_stats]
                
                logger.info(f"Data prepared - Openings: {len(openings)}, Wins: {len(wins)}, Draws: {len(draws)}, Losses: {len(losses)}")
                
                # Create DataFrame for plotting
                most_used_openings = pd.DataFrame({
                    'wins': wins,
                    'draws': draws,
                    'losses': losses
                }, index=openings)
                
                # Create figure with adjusted size ratio
                plt.figure(figsize=(15, 8))
                
                # Create unstacked bar plot
                ax = most_used_openings[["wins", "draws", "losses"]].plot.barh(
                    rot=0, 
                    color=["green", "blue", "red"], 
                    stacked=False
                )
                
                # Customize plot
                ax.set_facecolor('xkcd:white')
                
                # Add legend in a box
                ax.legend(
                    prop={'size': 14},
                    frameon=True,
                    facecolor='white',
                    edgecolor='black',
                    bbox_to_anchor=(1.0, 1.0),
                    loc='upper left',
                    borderaxespad=0.
                )
                
                # Center title with smaller size
                plt.title(title, size=20, y=1.02, pad=15, ha='center')
                
                # Adjust labels
                plt.xlabel("Number of Games", size=16, labelpad=20)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                
                # Add more space for the bars
                plt.subplots_adjust(left=0.3)
                
                # Save figure
                logger.info(f"Saving figure to: {output_path}")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Successfully created chart: {title}")
                
            except Exception as e:
                logger.error(f"Error creating chart {title}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                plt.close()
                raise
        
        # White openings analysis
        white_df = df[df['played_as'] == 'white']
        logger.info(f"Found {len(white_df)} white games")
        white_stats = calculate_opening_stats(white_df)
        
        # Create charts for most played openings (not most successful)
        create_opening_chart(
            sorted(white_stats, key=lambda x: x['total_games'], reverse=True),
            "Results for Top 5 Openings as White",
            os.path.join(output_dir, "result_top_5_wh.png"),
            best=True
        )
        
        # Black openings analysis
        black_df = df[df['played_as'] == 'black']
        logger.info(f"Found {len(black_df)} black games")
        black_stats = calculate_opening_stats(black_df)
        
        # Create charts for most played openings (not most successful)
        create_opening_chart(
            sorted(black_stats, key=lambda x: x['total_games'], reverse=True),
            "Results for Top 5 Openings as Black",
            os.path.join(output_dir, "result_top_5_bl.png"),
            best=True
        )
        
        logger.info("Top 5 openings analysis complete")
        
    except Exception as e:
        logger.error(f"Error in top 5 openings analysis: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise  # Re-raise to ensure the error is propagated

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

def wh_heatmap_beg(df: pd.DataFrame, username: str, vmax: Optional[float] = None) -> float:
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
    board = sns.heatmap(board_open, cmap='Reds', square=True, linewidths=.1, linecolor='black', vmax=vmax)
    board.set_title('Starting Square Heatmap as White', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_starting_white.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return board_open.values.max()

def wh_heatmap_end(df: pd.DataFrame, username: str, vmax: Optional[float] = None) -> float:
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
    board = sns.heatmap(board_open, cmap='Reds', square=True, linewidths=.1, linecolor='black', vmax=vmax)
    board.set_title('Landing Square Heatmap as White', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_landing_white.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return board_open.values.max()

def bl_heatmap_beg(df: pd.DataFrame, username: str, vmax: Optional[float] = None) -> float:
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
    board = sns.heatmap(board_df, cmap='Blues', square=True, linewidths=.1, linecolor='black', vmax=vmax)
    board.set_title('Starting Square Heatmap as Black', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_starting_black.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return board_matrix.max()

def bl_heatmap_end(df: pd.DataFrame, username: str, vmax: Optional[float] = None) -> float:
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
    board = sns.heatmap(board_df, cmap='Blues', square=True, linewidths=.1, linecolor='black', vmax=vmax)
    board.set_title('Landing Square Heatmap as Black', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_landing_black.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return board_matrix.max()

def create_combined_heatmaps(df: pd.DataFrame, username: str) -> None:
    """Create and combine all heatmaps"""
    try:
        logger.info("Creating combined heatmap visualizations...")
        
        # First pass to get max values for consistent scales
        white_start_max = wh_heatmap_beg(df, username, None)
        white_end_max = wh_heatmap_end(df, username, None)
        white_max = max(white_start_max, white_end_max)
        
        black_start_max = bl_heatmap_beg(df, username, None)
        black_end_max = bl_heatmap_end(df, username, None)
        black_max = max(black_start_max, black_end_max)
        
        # Second pass with consistent scales
        wh_heatmap_beg(df, username, white_max)
        wh_heatmap_end(df, username, white_max)
        bl_heatmap_beg(df, username, black_max)
        bl_heatmap_end(df, username, black_max)
        
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
            (most_used_bl, "top 3 black replies"),
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
