import methods as m
import json
import copy


#######################################################

# Get username and retrieve data via API
# Get rid of useless data

#username = "sudesh2911"
username = "tyrange"
#username = "tyrange"
all_games_List = m.getGames(username)
m.filterList(all_games_List, username)

#######################################################

# Import JSON openings file
# Build Decision Trees for Black and White Games

openings = json.load(open('openings2.json'))
WhiteTree = m.buildOpeningTree(openings)
BlackTree = copy.deepcopy(WhiteTree)

#######################################################

# Give the most played openings

opening_freq_white = copy.deepcopy(openings)
opening_freq_black = copy.deepcopy(openings)
for x in opening_freq_white:
    opening_freq_white[x] = 0
    opening_freq_black[x] = 0

# Feed all the Games in the Decision Tree

m.convertPGN(all_games_List, WhiteTree, BlackTree, opening_freq_white, opening_freq_black)

# Print Frequency of all Openings

for k,v in opening_freq_white.items():
    if v != 0:
        pass
        print(k,v)

for k,v in opening_freq_black.items():
    if v != 0:
        print(k,v)

#######################################################

# Returns how many times you have been in this position
# p = PGN converted into a list of moves

p = "e2e4 d7d5".split(" ")
w = WhiteTree.traverse(p, WhiteTree.root)
b = BlackTree.traverse(p, BlackTree.root)

print(w.attributes)
print(b.attributes)

#######################################################

