# Standard library imports
import os
from typing import Dict, Any

# Try to import third-party packages with helpful error messages
try:
    import streamlit as st
except ImportError:
    raise ImportError(
        "Streamlit is required. Install it with: pip install streamlit"
    )

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "Pandas is required. Install it with: pip install pandas"
    )

try:
    from PIL import Image
except ImportError:
    raise ImportError(
    "Pillow is required. Install it with: pip install Pillow"
)
# Local imports
import get_data as gd
import visualize as viz
import prediction as pred
try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import seaborn as sns
    import matplotlib.pyplot as plt
    from PIL import Image
    import chess.pgn
except ImportError as e:
    raise ImportError(f"""
    Missing required packages. Please install them using:
    pip install streamlit pandas plotly seaborn matplotlib pillow python-chess
    
    Error: {str(e)}
    """)

def check_dependencies() -> None:
    """Check if all visualization dependencies are installed"""
    required_packages = {
        'plotly': 'plotly',
        'seaborn': 'seaborn',
        'matplotlib': 'matplotlib',
        'chess': 'python-chess'
    }
    
    optional_packages = {
        'cairosvg': 'cairosvg'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
            
    if missing_packages:
        st.error(f"""
        Missing required packages: {', '.join(missing_packages)}
        Please install them using:
        pip install {' '.join(missing_packages)}
        """)
        st.stop()
        
    # Check optional packages
    missing_optional = []
    for package, pip_name in optional_packages.items():
        try:
            __import__(package)
        except (ImportError, OSError):
            missing_optional.append(pip_name)
            
    if missing_optional:
        st.warning(f"""
        Some optional features will be disabled. To enable all features, install:
        {' '.join(missing_optional)}
        
        For Windows users:
        1. Download GTK3 Runtime from:
           https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
        2. Run the installer
        3. Restart your Python environment
        """)

def init_session_state() -> None:
    """Initialize session state variables"""
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

def render_home_tab() -> None:
    """Render the home tab content"""
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        st.image("img_files/chess.png", width=50)
    with col2:
        st.title("Statistical Analysis of a Chess Player using Data Science Pipeline")
    with col3:
        st.image("img_files/chess.png", width=50)
        
    st.header("Introduction")
    st.write(
        "There is no tool available in the market which provides an in-depth analysis of "
        "a player's overall games. This software provides Chess players a tool to "
        "improve their chess game with assistance from several Machine Learning and "
        "Data Science techniques. It works by studying the player's previous games "
        "and deriving useful data to help them learn from their previous mistakes."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        render_tutorial()
    with col2:
        st.image("img_files/magnus.png")

def render_tutorial() -> None:
    """Render the tutorial section"""
    st.header("Tutorial")
    st.write(
        "Click on the tabs above to switch different modes of operation.\n\n"
        "1. User Input:\n"
        "   - Provides a complete statistical data analysis, based on your previous chess games\n"
        "   - Just enter your chess.com username and wait for a few seconds\n"
        "   - The program will download all your data via API\n\n"
        "2. Player Analysis:\n"
        "   - Provides a thorough analysis of your data with interactive charts\n"
        "   - Gives detailed instructions on how to interpret the charts\n\n"
        "3. Game Prediction:\n"
        "   - Enter you and your opponent's username\n"
        "   - See the prediction of your game outcome\n"
        "   - Uses Logistic Regression on previous game data"
    )

def render_user_input_tab() -> None:
    """Render the user input tab content"""
    st.title("User Input")
    
    username = st.text_input(
        "Enter your chess.com username:",
        placeholder="username"
    )
    
    if st.button("Request Analysis"):
        if username:
            with st.spinner("Fetching Data, Please Wait..."):
                try:
                    # Create progress bar
                    progress_text = "Operation in progress. Please wait."
                    progress_bar = st.progress(0, text=progress_text)
                    
                    # Ensure directory exists
                    os.makedirs(username, exist_ok=True)
                    
                    try:
                        # Fetch and process data
                        progress_bar.progress(25, text="Fetching games from Chess.com...")
                        gd.driver_fn(username)
                        
                        progress_bar.progress(50, text="Generating visualizations...")
                        viz.visualize_data(username)
                        
                        progress_bar.progress(100, text="Analysis complete!")
                        
                        # Update session state
                        st.session_state.username = username
                        st.session_state.analysis_complete = True
                        
                        # Check if key visualizations were created
                        required_files = ["top_op_wh.png", "corr_heatmap.png"]
                        missing_files = [f for f in required_files 
                                       if not os.path.exists(os.path.join(username, f))]
                        
                        if missing_files:
                            st.warning("Analysis complete but some visualizations could not be generated")
                        else:
                            st.success("Analysis Complete! Please proceed to the Player Analysis Tab.")
                            
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
                        st.info("Please make sure the username exists on chess.com and try again.")
                        return
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Please check your internet connection and try again.")
        else:
            st.warning("Please enter a username")

def render_analysis_tab() -> None:
    """Render the player analysis tab content"""
    st.title("Player Analysis")
    
    if st.session_state.analysis_complete and st.session_state.username:
        username = st.session_state.username
        if os.path.exists(f"{username}/corr_heatmap.png"):
            render_analysis_content(username)
    else:
        st.info("Please analyze a player in the User Input tab first.")

def render_analysis_content(username: str) -> None:
    """Render the analysis content for a given username"""
    try:
        # Top Openings as White
        st.header("Top 20 Most Played Openings as White")
        st.write("This is a frequency countplot chart of the user's top 20 most played openings as white.")
        
        top_op_wh_path = os.path.join(username, "top_op_wh.png")
        if os.path.exists(top_op_wh_path):
            st.image(top_op_wh_path)
        else:
            st.error("White openings analysis visualization not available")
            
        # Top Openings as Black
        st.header("Top 20 Most Played Openings as Black")
        st.write("This is a frequency countplot chart of the user's top 20 most played openings as black.")
        
        top_op_bl_path = os.path.join(username, "top_op_bl.png")
        if os.path.exists(top_op_bl_path):
            st.image(top_op_bl_path)
        else:
            st.error("Black openings analysis visualization not available")
        
        # Top First Moves
        st.header("Top 3 First Moves as White")
        col1, col2, col3 = st.columns(3)
        
        # Check for move visualizations or fallback to text file
        moves_text_path = os.path.join(username, "top_opening_moves.txt")
        if os.path.exists(moves_text_path):
            with open(moves_text_path, 'r') as f:
                moves = f.readlines()
                for i, move in enumerate(moves, 1):
                    st.write(f"Top Move {i}: {move.strip()}")
        else:
            # Try to display board images
            with col1:
                move1_path = os.path.join(username, "top_opening_move_as_white_1.png")
                if os.path.exists(move1_path):
                    st.image(move1_path)
            with col2:
                move2_path = os.path.join(username, "top_opening_move_as_white_2.png")
                if os.path.exists(move2_path):
                    st.image(move2_path)
            with col3:
                move3_path = os.path.join(username, "top_opening_move_as_white_3.png")
                if os.path.exists(move3_path):
                    st.image(move3_path)
        
        # Add heatmap visualizations
        st.header("Square Heatmaps")
        
        # White heatmaps
        white_heatmap_path = os.path.join(username, "heatmap_combined_white.png")
        if os.path.exists(white_heatmap_path):
            st.image(white_heatmap_path)
        else:
            st.warning("White square heatmaps not available")
            
        # Black heatmaps
        black_heatmap_path = os.path.join(username, "heatmap_combined_black.png")
        if os.path.exists(black_heatmap_path):
            st.image(black_heatmap_path)
        else:
            st.warning("Black square heatmaps not available")
        
        # Add other visualizations with existence checks
        for viz_file, title in [
            ("corr_heatmap.png", "Correlation Heatmap"),
            ("fight.png", "Game Length Analysis"),
            ("rating_ladder_red.png", "Rating Progress"),
            ("time_class.png", "Time Control Distribution"),
            ("result_as_wh.png", "Results as White"),
            ("result_as_bl.png", "Results as Black"),
            ("overall_results.png", "Overall Results"),
            ("result_top_5_wh.png", "Top 5 White Openings Results"),
            ("result_top_5_bl.png", "Top 5 Black Openings Results")
        ]:
            viz_path = os.path.join(username, viz_file)
            if os.path.exists(viz_path):
                st.header(title)
                st.image(viz_path)
            else:
                st.warning(f"{title} visualization not available")
                
    except Exception as e:
        st.error(f"Error rendering analysis content: {str(e)}")
        st.info("Some visualizations may not be available")

def render_prediction_tab() -> None:
    """Render the game prediction tab content"""
    st.title("Game Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        user1 = st.text_input(
            "Enter your username:",
            placeholder="your_username"
        )
    with col2:
        user2 = st.text_input(
            "Enter opponent's username:",
            placeholder="opponents_username"
        )
    
    if st.button("Predict Game Outcome"):
        if user1 and user2:
            with st.spinner("Building Logistic Regression Classifier Model..."):
                try:
                    results = pred.predict(user1, user2)
                    display_prediction_results(results)
                except Exception as e:
                    error_msg = str(e)
                    if "Could not find blitz rating" in error_msg:
                        st.error("⚠️ " + error_msg + "\nBoth players must have played blitz games on Chess.com to use this feature.")
                    elif "Error accessing Chess.com API" in error_msg:
                        st.error("🌐 Unable to access Chess.com API. Please check your internet connection and try again.")
                    elif "Error generating chess dataset" in error_msg:
                        st.error("📊 Error processing chess data. Please try running the analysis from the User Input tab first.")
                    else:
                        st.error("❌ " + error_msg)
        else:
            st.warning("Please enter both usernames")

def display_prediction_results(results: Dict[str, Any]) -> None:
    """Display the prediction results"""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Your Rating", results["user_rating"])
        st.metric("Rating Difference", results["rating_diff"])
    with col2:
        st.metric("Opponent's Rating", results["opp_rating"])
    
    with st.expander("View Model Details", expanded=False):
        st.code(results["summ1"], language="text")
        st.write(results["ord_acc"])
        st.write(results["cat_acc"])
    
    st.success(results["result"])

def render_about_tab() -> None:
    """Render the about tab content"""
    st.title("About")
    
    st.markdown(
        "> *I don't believe in psychology. I believe in good moves.*\n>\n> — Robert James Fischer",
        unsafe_allow_html=False
    )
    
    st.header("This software was written by:")
    st.subheader("Yogen Ghodke")
    
    st.header("Special Thanks to:")
    st.write("FreeCodeCamp, Corey Schaffer (YouTubers)")

def main() -> None:
    """Main application entry point"""
    # Check dependencies before starting the app
    check_dependencies()
    
    # Page config
    st.set_page_config(
        page_title="Chess Analysis Dashboard",
        page_icon="♟️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    init_session_state()

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            background-color: #262730;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0F1116;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create tabs
    tabs = st.tabs([
        "Home",
        "User Input",
        "Player Analysis",
        "Game Prediction",
        "About"
    ])

    # Render each tab
    with tabs[0]: render_home_tab()
    with tabs[1]: render_user_input_tab()
    with tabs[2]: render_analysis_tab()
    with tabs[3]: render_prediction_tab()
    with tabs[4]: render_about_tab()

if __name__ == "__main__":
    main()
