import os
import pandas as pd
import seaborn as sns
from typing import Dict, List

def wh_heatmap_end(df: pd.DataFrame, username: str) -> None:
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
        # if row["first_move"][0] == "a":
        #    a1[0] += 1

        di[row["first_move"][0] + "1"][int(row["first_move"][1]) - 1] += 1
        di[row["first_move"][2] + "2"][int(row["first_move"][3]) - 1] += 1

    row = ['1', '2', '3', '4', '5', '6', '7', '8']
    column = [None, None, None, None, None, None, None, None]

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

    board1 = sns.heatmap(board_open, cmap='Reds', square=True, linewidths=.1, linecolor='black')
    board1.set_title('Heatmap of Landing Square as White', size=18, y=1.05)

    board1.get_figure().savefig(username + "/heatmap_landing.png", bbox_inches='tight', dpi=300)


def driver_fn(username):
    import pandas as pd
    df = pd.read_csv(os.path.join(username, 'chess_dataset.csv'))
    wh_heatmap_end(df, username)


import sys

driver_fn(sys.argv[1])
