class Node:
    def __init__(self):
        self.data = None  # Nf3
        self.children = []  # [e2e4 , d2d4 , ...]
        self.attributes = \
            {
                "Opening Name": "-------", "Wins": 0, "Draws": 0, "Losses": 0, "Games": 0
            }
        self.hash = {}  # Children Keymapping {e2e4 : 0 , d2d4 : 1 ,...}


class Tree:
    def __init__(self):
        self.root = Node()
        self.root.data = "root"

    def builder(self, ptr, name, li):
        for x in li:
            if x in ptr.hash:
                ptr = ptr.children[ptr.hash[x]]
                continue
            else:
                ptr.hash.update({x: len(ptr.children)})
                ptr.children.append(Node())
                ptr = ptr.children[ptr.hash[x]]
                ptr.data = x
                continue
        ptr.attributes.update({"Opening Name": name})

    def traverse(self, li, ptr):
        for x in li:
            if x in ptr.hash:
                ptr = ptr.children[ptr.hash[x]]
            else:
                break
        return ptr

    def insertGames(self, li, ptr, result, freq):
        if result in ["win"]:
            result = "Wins"
        elif result in ["stalemate", "agreed", "", "repetition", "insufficient", "50move", "timevsinsufficient"]:
            result = "Draws"
        elif result in ["checkmated", "resigned", "timeout", "lose", "abandoned"]:
            result = "Losses"
        else:
            print("Invalid Result")

        open_name = ""
        for x in li:
            if x in ptr.hash:
                ptr = ptr.children[ptr.hash[x]]
                ptr.attributes.update({result: ptr.attributes[result] + 1, "Games": ptr.attributes["Games"] + 1})
                if ptr.attributes["Opening Name"] != "-------":
                    open_name = ptr.attributes["Opening Name"]
                continue
            else:
                freq[open_name] += 1
                break
            # else:
            #     ptr.hash.update({x: len(ptr.children)})
            #     ptr.children.append(Node())
            #     ptr = ptr.children[ptr.hash[x]]
            #     ptr.data = x
            #     ptr.attributes.update({result: ptr.attributes[result] + 1, "Games": ptr.attributes["Games"] + 1})
            #     continue

    def checkNextData(self, node, d, ptr):
        flag = False
        for x in node.children:
            if x.data == d:
                ptr = x
                flag = True
        return flag
