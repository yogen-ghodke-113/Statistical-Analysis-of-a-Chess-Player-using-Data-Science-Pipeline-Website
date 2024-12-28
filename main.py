import streamlit as st
import io
import logging
import time

# Must be the first Streamlit command
st.set_page_config(
    page_title="Chess Analysis Dashboard",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

try:
    import cairosvg
except ImportError:
    raise ImportError(
        "CairoSVG is required. Install it with: pip install cairosvg"
    )

def check_dependencies() -> None:
    """Check if all visualization dependencies are installed"""
    required_packages = {
        'plotly': 'plotly',
        'seaborn': 'seaborn',
        'matplotlib': 'matplotlib',
        'chess': 'python-chess'
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

def init_session_state() -> None:
    """Initialize session state variables"""
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

def render_home_tab() -> None:
    """Render the home tab content"""
    # Adjust column ratios for better alignment
    col1, col2, col3 = st.columns([0.5, 6, 0.5])
    
    # Add CSS for vertical alignment
    st.markdown("""
        <style>
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            padding-top: 15px;  /* Adjust this value to align with title */
        }
        </style>
    """, unsafe_allow_html=True)
    
    with col1:
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image("img_files/logo.ico", width=40)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <h1 style='text-align: center; color: white; margin: 0;'>
                Statistical Analysis of a Chess Player using Data Science Pipeline
            </h1> <hr>
            """, 
            unsafe_allow_html=True
        )
    with col3:
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image("img_files/logo.ico", width=40)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.header("Introduction")
    st.write(
        "There is no tool available in the market which provides an in-depth analysis of "
        "a player's overall games. This software provides Chess players a tool to "
        "improve their chess game with assistance from several Machine Learning and "
        "Data Science techniques. It works by studying the player's previous games "
        "and deriving useful data to help them learn from their previous mistakes."
    )
    
    # Give more space to the images column
    col1, col2 = st.columns([1.5, 1])
    with col1:
        render_tutorial()
    with col2:
        # Add custom CSS to increase image size
        st.markdown("""
            <style>
            .stImage > img {
                max-width: 100%;
                height: auto;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Display Magnus
        st.image("img_files/magnus.png", use_container_width=True)
        
        # Create nested columns with more width
        st.markdown("<div style='padding: 10px 0px;'></div>", unsafe_allow_html=True)  # Add spacing
        subcol1, subcol2 = st.columns([1, 1])
        with subcol1:
            st.image("img_files/bobby.jpg", use_container_width=True)
        with subcol2:
            st.image("img_files/kasparov.png", use_container_width=True)

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

def check_cached_analysis(username: str) -> bool:
    """Check if all required files exist in the player's cache directory"""
    required_files = [
        "chess_dataset.csv",
        "chess_dataset_adv.csv",
        "result_as_wh.png",
        "result_as_bl.png",
        "fight.png",
        "rating_ladder_red.png",
        "time_class.png",
        "overall_results.png",
        "result_top_5_wh.png",
        "result_top_5_bl.png",
        "corr_heatmap.png"
    ]
    
    player_dir = os.path.join('player_data', username)
    if not os.path.exists(player_dir):
        return False
        
    return all(os.path.exists(os.path.join(player_dir, file)) for file in required_files)

def render_user_input_tab() -> None:
    """Render the user input tab content"""
    st.title("User Input")
    
    username = st.text_input(
        "Enter your Chess.com username:",
        placeholder="your_username"
    )
    
    if st.button("Analyze Games"):
        if username:
            # Create placeholder for progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_output = st.empty()
            
            try:
                # Update status
                status_text.text("🔄 Downloading games from Chess.com...")
                progress_bar.progress(10)
                
                # Create StringIO to capture logs
                log_capture = io.StringIO()
                log_handler = logging.StreamHandler(log_capture)
                log_handler.setFormatter(logging.Formatter('%(message)s'))
                logging.getLogger().addHandler(log_handler)
                
                # Download data
                gd.driver_fn(username)
                progress_bar.progress(40)
                status_text.text("🔄 Processing game data...")
                
                # Update log display
                log_output.code(log_capture.getvalue())
                
                # Visualize data
                status_text.text("🔄 Creating visualizations...")
                progress_bar.progress(70)
                viz.visualize_data(username)
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Update session state
                st.session_state.username = username
                st.session_state.analysis_complete = True
                
                # Show success message with instructions
                st.success("""
                ✅ Analysis complete! 
                
                Click on the Player Analysis tab to view your results.
                """)
                
                # Keep success message visible
                time.sleep(2)
                
            except Exception as e:
                progress_bar.empty()
                error_msg = str(e)
                if "User not found" in error_msg:
                    st.error("❌ " + error_msg)
                elif "No games found" in error_msg:
                    st.error("❌ " + error_msg)
                elif "Unable to connect" in error_msg:
                    st.error("🌐 " + error_msg)
                else:
                    st.error("❌ An error occurred during analysis: " + error_msg)
            finally:
                # Remove log handler
                logging.getLogger().removeHandler(log_handler)
        else:
            st.warning("Please enter a username")

def render_analysis_tab() -> None:
    """Render the player analysis tab content"""
    st.title("Player Analysis")
    
    if st.session_state.analysis_complete and st.session_state.username:
        username = st.session_state.username
        if os.path.exists(os.path.join('player_data', username, "corr_heatmap.png")):
            render_analysis_content(username)
    else:
        st.info("Please analyze a player in the User Input tab first.")

def render_analysis_content(username: str) -> None:
    """Render the analysis content for a given username"""
    try:
        # Top Openings as White
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Top 20 Most Played Openings as White")
        st.write("This is a frequency countplot chart of the user's top 20 most played openings as white.")
        
        top_op_wh_path = os.path.join('player_data', username, "top_op_wh.png")
        if os.path.exists(top_op_wh_path):
            st.image(top_op_wh_path)
        else:
            st.error("White openings analysis visualization not available")
            
        # Top Openings as Black
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Top 20 Most Played Openings as Black")
        st.write("This is a frequency countplot chart of the user's top 20 most played openings as black.")
        
        top_op_bl_path = os.path.join('player_data', username, "top_op_bl.png")
        if os.path.exists(top_op_bl_path):
            st.image(top_op_bl_path)
        else:
            st.error("Black openings analysis visualization not available")
        
        # Top First Moves
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Top 3 First Moves as White")
        cols = st.columns([1.2, 1.2, 1.2, 0.1])
        
        # Check for move visualizations
        for i, col in enumerate(cols[:-1], 1):
            svg_path = os.path.join('player_data', username, f'top_opening_move_as_white_{i}.svg')
            png_path = os.path.join('player_data', username, f'top_opening_move_as_white_{i}.png')
            
            with col:
                if os.path.exists(svg_path):
                    try:
                        # Convert SVG to PNG if not already done
                        if not os.path.exists(png_path):
                            cairosvg.svg2png(
                                url=svg_path,
                                write_to=png_path,
                                scale=2.0  # Increase quality
                            )
                        # Display PNG image
                        st.image(png_path)
                    except Exception as e:
                        st.error(f"Error converting chess board {i}: {str(e)}")
                else:
                    st.warning(f"Move {i} visualization not available")
        
        # Top Black Replies
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Top 3 Replies as Black")
        cols = st.columns([1.2, 1.2, 1.2, 0.1])
        
        # Check for black reply visualizations
        for i, col in enumerate(cols[:-1], 1):
            svg_path = os.path.join('player_data', username, f'top_reply_move_as_black_{i}.svg')
            png_path = os.path.join('player_data', username, f'top_reply_move_as_black_{i}.png')
            
            with col:
                if os.path.exists(svg_path):
                    try:
                        # Convert SVG to PNG if not already done
                        if not os.path.exists(png_path):
                            cairosvg.svg2png(
                                url=svg_path,
                                write_to=png_path,
                                scale=2.0  # Increase quality
                            )
                        # Display PNG image
                        st.image(png_path)
                    except Exception as e:
                        st.error(f"Error converting chess board {i}: {str(e)}")
                else:
                    st.warning(f"Move {i} visualization not available")
        
        # Add heatmap visualizations
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Heatmaps of Starting Moves as White and Black")
        st.write("These are the heatmap of the first moves made as white and black. Darker squares represent higher frequency.")
        
        # White heatmaps
        white_heatmap_path = os.path.join('player_data', username, "heatmap_combined_white.png")
        if os.path.exists(white_heatmap_path):
            st.image(white_heatmap_path)
        else:
            st.warning("White square heatmaps not available")
            
        # Black heatmaps
        black_heatmap_path = os.path.join('player_data', username, "heatmap_combined_black.png")
        if os.path.exists(black_heatmap_path):
            st.image(black_heatmap_path)
        else:
            st.warning("Black square heatmaps not available")
        
        # Results by Color (side by side)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Results by Color")
        st.write("These are the results of the games played by the user.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Results as White")
            st.write("This donut chart shows your win/draw/loss ratio when playing as white.")
            white_results_path = os.path.join('player_data', username, "result_as_wh.png")
            if os.path.exists(white_results_path):
                st.image(white_results_path)
            else:
                st.warning("White results visualization not available")
                
        with col2:
            st.subheader("Results as Black")
            st.write("This donut chart shows your win/draw/loss ratio when playing as black.")
            black_results_path = os.path.join('player_data', username, "result_as_bl.png")
            if os.path.exists(black_results_path):
                st.image(black_results_path)
            else:
                st.warning("Black results visualization not available")
        
        # Add other visualizations with existence checks
        visualizations = [
            ("fight.png", "How much of a fight the user puts up when losing", 
             "These are all the games where the user lost. More number of moves in the games means the user put up a good fight before resigning. Less number of moves indicate that the player blundered early on in the game."),
            ("time_class.png", "Time Control Distribution", 
             "This pie chart shows the distribution of different time controls in your games."),
            ("rating_ladder_red.png", "Rating Progress", 
             "This chart shows your rating progression over your last 150 rated games in different time controls."),
            ("overall_results.png", "Overall Results", 
             "A Frequency plot of the result of all the games, the user has played on the website."),
            ("overall_results_pie.png", "Overall Results Distribution (Pie Chart)", 
             "A pie chart showing the distribution of all game results."),
            ("result_top_5_wh.png", "Opening Strength and Weakness Analysis as White", 
             "These graphs are very important for Strength / Weakness Analysis. Longer Red bar indicates the opening played by the user the most, but also lost the most. Longest green bar indicates the strongest most played opening."),
            ("result_top_5_bl.png", "Opening Strength and Weakness Analysis as Black", 
             "These graphs are very important for Strength / Weakness Analysis. Longer Red bar indicates the opening played by the user the most, but also lost the most. Longest green bar indicates the strongest most played opening."),
            ("corr_heatmap.png", "Correlation Heatmap", 
             "This heatmap shows correlations between different numerical aspects of your games.")
        ]
        
        for viz_file, title, description in visualizations:
            viz_path = os.path.join('player_data', username, viz_file)
            if os.path.exists(viz_path):
                st.markdown("<hr>", unsafe_allow_html=True)  # Add horizontal rule before heading
                st.header(title)
                st.write(description)
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
            # First check if the advanced dataset exists
            adv_dataset_path = os.path.join('player_data', user1, 'chess_dataset_adv.csv')
            if not os.path.exists(adv_dataset_path):
                st.error("📊 Error processing chess data. Please try running the analysis from the User Input tab first.")
                return
                
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
        st.metric("Opponent Rating", results["opp_rating"])
    
    with st.expander("View Model Details", expanded=False):
        st.code(results["summ1"], language="text")
        st.write(results["ord_acc"])
    
    st.success(results["result"])

def render_about_tab() -> None:
    """Render the about tab content"""
    st.title("About")
    
    # Create two columns with different widths
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
            #### © Yogen Ghodke
            #### Contact Information
            - **LinkedIn**: [Yogen Ghodke](https://www.linkedin.com/in/yogenghodke/)
            - **Email**: yogenghodke@gmail.com
        """)
        
        # Add resume display with a button
        resume_path = "img_files/Yogen_Ghodke_Resume_28_Nov_24.pdf"
        if os.path.exists(resume_path):
            with open(resume_path, "rb") as pdf_file:
                PDFbyte = pdf_file.read()
            st.download_button(
                label="📄 Download Resume",
                data=PDFbyte,
                file_name="Yogen_Ghodke_Resume.pdf",
                mime="application/pdf"
            )
        
        st.markdown("""
            ### I'm currently open to job opportunities! Feel free to reach out via LinkedIn or email.
            ---
            ### Special Thanks
            - **FreeCodeCamp**
              - For their comprehensive tutorials and resources
            - **Corey Schafer**
              - For excellent Python programming tutorials
            - **Chess.com**
              - For providing the API and game data
        """)
    
    with col2:
        # Add Gukesh's image and congratulatory message
        st.image("img_files/gukesh.jpg", width=450)
        st.markdown("""
            <p style='text-align: center; font-style: italic; font-size: 1.2em; margin-top: 10px;'>
                Congratulations to Gukesh for becoming the youngest World Chess Champion!
            </p>
        """, unsafe_allow_html=True)

def main() -> None:
    """Main application entry point"""
    # Check dependencies before starting the app
    check_dependencies()
    
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
