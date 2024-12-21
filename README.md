# Statistical Analysis of a Chess Player using Data Science Pipeline

## About The Project

Chess is a game with infinite possibilities and every game played produces large amounts of data. This project aims to provide chess enthusiasts a tool to improve their game with assistance from several Machine Learning and Data Science techniques, by studying their previous games.

## Objectives
* Finding the player’s Strengths and Weaknesses.
* Detailed analysis of your opponents moves before the match.
* Next move Predictions based on previous games played.
* See the least and most favourite Chess Opening Lines.
* Find which moves attributed to Wins , Losses or Draws.
* Visualising the statistics related to the account in great detail.
* Help the player improve their endgames.
* Play a Live game and get the analysis of the game in real-time.

## Navigation

* This is the screenshot of the Home Page of the App. On the top are the tabs which the user can click to access different context functions in the App.
<br>
<p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/readme_files/home.png">
  </a>
  </p>
 <br>


* For a complete Data Analysis, you must go to User Input tab and enter your Chess.com username and click on Request Analysis. The application will take some time to retrieve your data through the Chess.com API.

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/readme_files/userinput.png">
  </a>
  </p><br>
  
* Once done, you will receive a dialog saying successful. Click on Ok and go to the Player Analysis Tab. Here you will see all the visualized data in detail.

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/readme_files/playeranalysis.png">
  </a>
  </p><br>
  
<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/readme_files/topmoves.png">
  </a>
  </p> <br>
  
<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/readme_files/strengthsweakness.png">
  </a>
  </p> <br>
  
* To predict what is the probability that one of the two players will win against the other, go to the game predictions tab. Here, enter you and your opponent's Chess.com username and click on the request analysis button. This will build a logistic regression model based on both the players' historical wins against opponents of different ELO ratings. Once done, you can scroll down to see the probability of winning and the accuracy of the model. Note that, more the games played on your Chess.com account, more will be the model's accuracy, as more data will be available for an accurate prediction.

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/readme_files/gameprediction.png">
  </a>
  </p> <br>

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/readme_files/regression.png">
  </a>
  </p> <br>
 
 <br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/readme_files/predictionresult.png">
  </a>
  </p> <br>
 
* This is the about page giving credit to people who really helped me out with their tutorials.

 <br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/readme_files/about.png">
  </a>
  </p> 
<br>

## The following are the different Types of Analysis available :

### Frequency Countplot of the user's top 20 different openings played as White

<br> <p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/top_op_wh.png">
  </a>
  </p> <br>
  
### Frequency Countplot of the user's top 20 most played defences played as Black in response to white's first move.

 <br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/top_op_bl.png">
  </a>
  </p> <br>

### Top Opening Move as White

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/top_opening_move_as_white_1.png">
  </a>
  </p> <br>
  
### HeatMaps of Starting Squares as White

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/heatmap_starting.png">
  </a>
  </p> <br>
  
### HeatMaps of Landing Squares as White
  
 <br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/heatmap_landing.png">
  </a>
  </p> <br>

### Correlation Heatmap of all the features of the downloaded dataset. 

Higher number means higher correlation.

  
 <br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/corr_heatmap.png">
  </a>
  </p> <br>
  
### Player's Elo Rating in the last 150 Rated Games

  
 <br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/rating_ladder_red.png">
  </a>
  </p> <br>
  
### Types of different Time Class Control Games played. 

Different Classes of time control are blitz, bullet, classical, puzzle, chess960 and rapid.

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/time_class.png">
  </a>
  </p> <br>
  
### How much of a fight the user puts up when losing.
 
These are all the games where the user lost. More number of moves in the games means the user put up a good fight before resigning. Less number of moves indicate that the player blundered early on in the game.

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/fight.png">
  </a>
  </p> <br>
  
### A Frequency plot of the result of all the games, the user has played on the website.

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/overall_results.png">
  </a>
  </p> <br>
  
### A pie chart of the result of all the games, the user has played on the website.

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/result_pi.png">
  </a>
  </p> <br>
  
### Donut Chart of percentage Wins, Losses and Draws as White

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/result_as_wh.png">
  </a>
  </p> <br>
  
### Donut Chart of percentage Wins, Losses and Draws as Black
<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/result_as_bl.png">
  </a>
  </p> <br>
  
  
### Strength and Weakness Analysis

These graphs are very important for Strength / Weakness Analysis. Longer Red bar indicates the opening played by the user the most, but also lost the most. Longest green bar indicates the strongest most played opening.
  
### Top 5 Strengths / Weaknesses as White :

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/result_top_5_wh.png">
  </a>
  </p> <br>

### Top 5 Strengths / Weaknesses as Black :

<br><p align="center">
  <a href="https://github.com/yogen-ghodke-113/Statistical-Analysis-of-a-Chess-Player-using-Data-Science-Pipeline">
    <img src="/tyrange/result_top_5_bl.png">
  </a>
  </p> <br><br>
	


### Built With
The following libraries and frameworks were used in the making of this Project.
* [Python Imaging Library (PIL)](https://pypi.org/project/Pillow/)
* [CairoSVG](https://pypi.org/project/CairoSVG/)
* [Tkinter](https://docs.python.org/3/library/tkinter.html)
* [ZipFile](https://docs.python.org/3/library/zipfile.html)
* [Python Requests Module](https://docs.python-requests.org/)
* [Pandas](https://pandas.pydata.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Plotly](https://plotly.com/)
* [Python Chess Library](https://python-chess.readthedocs.io/en/latest/pgn.html)
* [Scikit Learn](https://scikit-learn.org/)
* [Python Mord](https://pypi.org/project/mord/)



