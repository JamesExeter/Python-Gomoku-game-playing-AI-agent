import numpy as np
import copy
from misc import legalMove
from gomokuAgent import GomokuAgent


# Node class holding information.
class Node:
    def __init__(self, board, ancestor=None, children=None, value=None, coordinates=None):
        self.board = board
        self.ancestor = ancestor
        self.children = children
        self.value = value
        self.coordinates = coordinates


# Some sort of heuristic attempt. Doesn't really work.
def evaluate_board(board, player_id):
    score = 0

    position = np.where(board == player_id)
    player_coordinates = list(zip(position[0], position[1]))

    for coord in player_coordinates:
        for x in range(-1, 1):
            for y in range(-1, 1):
                try:
                    if coord[0+x] and coord[1+y] == player_id:
                        score += 100
                except:
                    continue

    return score


# Minimax function that evaluates the leaf nodes and passes up
# the values through the tree.
def minimax(node, depth, max_player, player_id):
    if depth == 0:
        node.value = evaluate_board(node.board, player_id)
        return node.value

    if max_player:
        value = float('-inf')
        for node in node.children:
            value = max(value, minimax(node, depth - 1, False, player_id))
            node.value = value
        return value

    else:
        value = float('inf')
        for node in node.children:
            value = min(value, minimax(node, depth - 1, True, player_id))
            node.value = value
        return value


# Returns a list of all possible moves, made by any given player
# on any given board.
def precognition(player_id, board, node):

    returning_nodes = []

    positions = np.nonzero(board == player_id)

    coordinates = list(zip(positions[0], positions[1]))

    for coord in coordinates:
        for x_pos in range(-1, 1):
            for y_pos in range(-1, 1):

                location = (coord[0] + x_pos, coord[1] + y_pos)

                if legalMove(board, location):

                    new_board = copy.deepcopy(board)

                    new_board[location[0]][location[1]] = player_id

                    new_node = Node(new_board, node, coordinates=location)

                    returning_nodes.append(new_node)

    return returning_nodes


# Uses the precognition function to create a tree of all moves, shifting between each player.
# Hardcoded for a depth of 3 currently.
def generate_tree(player_id, parent_node):
    if player_id == -1:

        next_moves = precognition(-1, parent_node.board, parent_node)
        parent_node.children = next_moves

        for new_node in next_moves:

            further_moves = precognition(1, new_node.board, new_node)
            new_node.children = further_moves

            for further_node in further_moves:

                even_future_moves = precognition(-1, further_node.board, further_node)
                further_node.children = even_future_moves

    else:

        next_moves = precognition(1, parent_node.board, parent_node)
        parent_node.children = next_moves

        for new_node in next_moves:

            further_moves = precognition(-1, new_node.board, new_node)
            new_node.children = further_moves

            for further_node in further_moves:
                even_future_moves = precognition(1, further_node.board, further_node)
                further_node.children = even_future_moves


class Player(GomokuAgent):
    def move(self, board):

        while True:

            move_loc = tuple(np.random.randint(self.BOARD_SIZE, size=2))

            current_state = Node(board, ancestor=False)

            generate_tree(self.ID, current_state)

            best_value = minimax(current_state, 3, True, self.ID)

            for child in current_state.children:
                if child.value == best_value:
                    move_loc = child.coordinates

            if legalMove(board, move_loc):
                return move_loc


"""
Code that would have been used to score possible moves
Upon placing a new move, the representation of all connections
for the board is updated using the check_neighbours function and all of the code following it
the board for each opponent will then comprise of a list of all 2+ length connections denoting the 
following information: 1) the coordinates of the connections 2) the length of the connections
3) the direction the connection is going in and 4) how many blocked ends are on the connection

It was hoped that using this information, we could create a score of all possible moves, taking into account
when to be offensive and when to be defernsive i.e., if an opponent has any 3 connections, we should seek
to block them off a.s.a.p, but they did not, we should focus on expanding our connections whilst limiting theirs

# all the directions a move can be made in, move classified into index too
directions = [("U", 0), ("D", 1), ("L", 2), ("R", 3), ("UL", 4), ("UR", 5), ("DL", 6), ("DR", 7)]
# all of the chronological moves the opponent has made
opponent_moves = []
# all of the chronological moves we have made
our_moves = []
# the last move our opponent made
last_op_move = None
# the last move we made
last_move_ours = None
# stores the compressed version of the opponent moves
opponent_analysis = []
# stores the compressed version of our moves
player_analysis = []
# constructor to the parent class
ID = None

ATTACK_VALUE = 1
DEFENSE_VALUE = 1

# querying the direction for the move vector
# e.g. if we have [True, True, True, False, True, False, True, False]
# when we index the directions, we know a move in the right direction is not valid to make
def check_valid(move, move_vectors):
    return move_vectors[move[1]]

# implemented switch statement, given direction, return the alteration it makes to a current position
def neighbour_caser(direction):
    switcher = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
        4: (-1, -1),
        5: (-1, 1),
        6: (1, -1),
        7: (1, 1)
    }
    return switcher.get(direction, "Invalid direction")

# switch statement function that takes a direction and outputs the opposite direction
def opposite_direction(direction):
    switcher = {
        0 : 1,
        1 : 0,
        2 : 3,
        3 : 2,
        4 : 7,
        5 : 6,
        6 : 5,
        7 : 4
    }
    return switcher.get(direction, "Invalid direction")

# finds the last move made by a player once given the most current board and also returns the updated list of all moves
def find_last(moves_arr, symbol, board_size, current):
    # this is the case where we are making the first move
    if current is np.zeros((board_size, board_size), dtype=int):
        # there can be previous move
        return None, moves_arr
    
    # scan the board for any changes in the board for a selected player
    for i in range(board_size):
        for j in range(board_size):
            # checks to make sure that move has not be recorded yet
            if (current[i][j] == symbol) and ((i, j) not in moves_arr):
                moves_arr.append((i, j))
                return moves_arr[-1], moves_arr

    # shouldn't ever be reached
    # this is where the turn player has yet to make a move
    return None, moves_arr

# only return the board analysis items greater than a certain length
# e.g. if length 2 given, only return connected positions of 3+
def filter_representation(analysis, length):
    return [item for item in analysis if item[1] > length]

def filter_blocked(connections):
        return [x for x in connections if x[3] > 0]

def update_blocked_score(analysis, left, right):
    if left is None:
        analysis[3] = 1
    if right is None:
        analysis[3] = 1
    if left is None and right is None:
        analysis[3] = 0

def check_in_board(move, board_size):
    board_size = board_size[0]
    if (0 <= move[0] < board_size) and (0 <= move[1] < board_size):
        return True
    
    return False

# given a connected block of counters, check if it has been blocked off on both sides
def check_blocked(analysis, check_left, invert_dir, current):
    pos = None

    if check_left:
        pos = analysis[0][0]
    else:
        pos = analysis[0][-1]

    # find the direction the connected counters are in
    dir_to_check = analysis[2][1]
    if invert_dir:
        dir_to_check = opposite_direction(dir_to_check)

    # check current side of array
    # check not at board edge for that direction
    neighbour = neighbour_caser(dir_to_check)
    new_pos_x, new_pos_y = pos[0] + neighbour[0], pos[1] + neighbour[1]

    if not check_in_board((new_pos_x, new_pos_y), current.shape):
        return True, None
    else:
        # find location on board where the connected pieces end that need to be blocked
        new_pos_x, new_pos_y = pos[0] + neighbour[0], pos[1] + neighbour[1]

        if (new_pos_x, new_pos_y) in analysis[0]:
            return None, None

        # if location contains a 0, then it is not blocked, else it is blocked
        if current[new_pos_x][new_pos_y] == 0:
            return False, (new_pos_x, new_pos_y)
        else:
            return True, None

# given a move, find for that move, all the valid moves that can be made
# valid in this case being moves that stay inside the board
def move_vectors(move, current):
    move_vectors = []
    # permuate all of the surrounding board elements
    for x, y in [(-1,0), (1,0), (0,-1), (0, 1), (-1,-1), (-1,1), (1, -1), (1, 1)]:
        move_vectors.append(check_in_board((move[0]+x, move[1]+y), current.shape))
    
    # each returned true or also value refers to whether a direction is valid to move in
    # since each direction also contains an index, it is easy to query this result later
    return move_vectors

def get_blocked_connection(analysis, check_left, invert_dir, current):
    out_arr = []

    if len(analysis) == 0:
        return []

    for elem in analysis:
        left, pos_l = check_blocked(elem, check_left, invert_dir, current)
        if left is None:
            left, pos_l = check_blocked(elem, check_left, not invert_dir, current)
        right, pos_r = check_blocked(elem, not check_left, invert_dir, current)
        if right is None:
            right, pos_r = check_blocked(elem, not check_left, not invert_dir, current)
        
        if not (pos_l is None):
            out_arr.append(pos_l)

        if not (pos_r is None):
            out_arr.append(pos_r)  

        update_blocked_score(elem, pos_l, pos_r)
    
    return out_arr

def check_neighbours(symbol, current, analysis, future_move=None):
    move = None
    if not (future_move is None):
        move = future_move
    else:
        move = move_to_use(symbol)

    # determine which player's last move we are using
    move = move_to_use(symbol)
    # for that last move, find the directions in we can check that are in the board
    moves = move_vectors(move, current)

    # we iterate through every direction
    for direction in directions[::2]:
        # check if last move is connected to anything
        # returns a position of a neighbouring connected piece, else if none, return none
        pos = find_neighbour(direction, symbol, move[0], move[1], moves, current)
        update_representation(symbol, direction, move, pos, moves, current, analysis)

# similar to representation_to_update function but returns the last move to use
def move_to_use(symbol):
    if symbol == ID:
        return last_move_ours
    
    return last_op_move

def order_based_on_direction(direction):
    #if 1, order ascending, if -1, ordering descending
    switcher = {
        0 : (-1, 0),
        1 : (1, 0),
        2 : (-1, 1),
        3 : (1, 1),
        4 : (-1, 1),
        5 : (-1, 1),
        6 : (1, 1),
        7 : (1, 1)
    } 
    return switcher.get(direction, "Invalid direction")

def order_connection_list(direction, analysis):
    order, dim = order_based_on_direction(direction[1])
    list(set(analysis))
    if order == 1:
        return analysis.sort(key=itemgetter(dim))
        
    return analysis.sort(key=itemgetter(dim), reverse=True)

# given a direction, the move we are checking and all of the valid moves, we check if that moves has any
# adjacent connections
def find_neighbour(direction, symbol, last_x, last_y, move_vectors, current, is_empty=None):
    # first check move is valid for that direction
    valid = check_valid(direction, move_vectors)
    if valid:
        # for a valid direction, get the alteration to that move
        move_tup = neighbour_caser(direction[1])
        # calculate the new position from our move in the given direction
        new_pos_x, new_pos_y = last_x + move_tup[0], last_y + move_tup[1]
        # we are finding if there is a neighbouring piece, if so, return the piece and then find the array it is in
        # if there is no array with that neighbour, we shall create a new one with a length of two, else increment the length by 1
        if is_empty:
            if (current[new_pos_x][new_pos_y] == 0):
                return (new_pos_y, new_pos_y)
        else:
            if (current[new_pos_x][new_pos_y] == symbol):
                return (new_pos_x, new_pos_y)
    
    #  Not valid to move in that direction or neighbour not what we are looking for
    return None

#  used to remove any duplicates from any of the connections
def check_connection_duplicates(analysis, original):
    to_remove = []

    # only check the elements of length 3 and over
    # generates all possible pairs of length 3 and 4
    list_of_pairs = [(analysis[a1], analysis[a2]) for a1 in range(len(analysis)) for a2 in range(a1+1,len(analysis))]
    for i in range(len(list_of_pairs)):
        # for each pair, if they have the same elements, then there is a duplicate, add one of the items to be removed
        if ((list_of_pairs[i][0][0] == list_of_pairs[i][1][0][::-1]) or (list_of_pairs[i][0][0] == list_of_pairs[i][1][0])):
            to_remove.append(list_of_pairs[i][0])
    
    for m in original:
        for n in original:
            if set(m[0]).issubset(set(n[0])) and m[0] != n[0]:
                to_remove.append(m)
                break

    # for each item to remove, remove it from the original list
    for item in to_remove:
        if item in original:
            original.remove(item)

    return original

def gen_neighbours(symbol, move, current, future_move=None):
    out_moves = []
    moves = move_vectors(move, current)

    for direction in directions:
        pos = find_neighbour(direction, symbol, move[0], move[1], moves, current, True)
        if not pos is None:
            out_moves.append(pos)
    
    return out_moves

def create_two_length(symbol, direction, move, pos, analysis, current):
    # need to check board for single standalone values
    # same idea as above, but initialise length to 2
    new_connection = None
    if current[pos[0]][pos[1]] == symbol:
        new_connection = [[move, pos], 2, direction, 2]
        order_connection_list(direction, new_connection[0])

    return new_connection

# updates the list of connections for a specified player
#takes the player symbol, the direction to check in, a move, a valid neighbour to connect to
#an array storing whether for each direction, we stay on the board for that move, the current board and the connection representation to update
def update_representation(symbol, direction, move, pos, move_vectors, current, analysis):
    added = None
    last_updated = None

    if not (pos is None):
        #only needs to be initialised once since can only be added to one array
        added = False
        for item in analysis:
            # only append to existing list if in the same direction / opposite direction
            if (direction[1] == item[2][1]) or (item[2][1] == opposite_direction(direction[1])):
                if pos in item[0] and not (move in item[0]):
                    # update the length of the list
                    # and add the item
                    item[0].append(move)
                    item[1] = len(item[0])
                    order_connection_list(item[2], item[0])
                    added = True
                    last_updated = item

        if not added:
            last_updated = create_two_length(symbol, direction, move, pos, analysis, current)
            analysis.append(last_updated)
    
    #check for opposite direction
    op_dir = directions[opposite_direction(direction[1])]
    opposite_pos = find_neighbour(op_dir, symbol, move[0], move[1], move_vectors, current)
    added = False
    if not (opposite_pos is None):
        for item in analysis:
            if (direction[1] == item[2][1]) or (item[2][1] == opposite_direction(direction[1])):
                if opposite_pos in item[0]:
                    #add each element from that array to the current one
                    if last_updated is None:
                        item[0].append(move)
                        item[1] = len(item[0])
                        order_connection_list(item[2], item[0])
                        added = True
                    else:
                        for loc in item[0]:
                            if loc not in item[0]:
                                last_updated[0].append(loc)

                        last_updated[1] = len(last_updated[0])
                        added = True
                        order_connection_list(last_updated[2], last_updated[0])
                
        if not added:
            if last_updated is None:
                last_updated = create_two_length(symbol, direction, move, opposite_pos, analysis, current)
                analysis.append(last_updated)
            else:
                last_updated[0].append(opposite_pos)
                last_updated[1] = len(last_updated[0])
                order_connection_list(last_updated[2], last_updated[0])

class Node:
    def __init__(self, board, size, ppa=None, poa=None,  ancestor=None, children=None, terminal=False, value=None, coordinates=None):
        self.board = board
        self.BOARD_SIZE = size
        self.ancestor = ancestor
        self.children = children
        self.terminal = terminal
        self.value = value
        self.coordinates = coordinates
        # all of the chronological moves the opponent has made
        self.opponent_moves_copy = []
        # all of the chronological moves we have made
        self.our_moves_copy = []
        # the last move our opponent made
        self.last_op_movecopy_ = None
        # the last move we made
        self.last_move_ours_copy = None
        # stores the compressed version of the opponent moves
        self.opponent_analysis_copy = poa
        # stores the compressed version of our moves
        self.player_analysis_copy = ppa
        # constructor to the parent class

#scored individual moves based off of the lengths of connections
#in the board, if the opponent has any length 3+, we aim to value that incredibly highly
#we use a modifiable ATTACK and DEFENCE value that picks whether to be offensive or defensive
def valCalc(node):
    valsum = 0
    #print("Our Analysis:\n ", node.player_analysis_copy)
    for i in node.player_analysis_copy:
        if i[1] == 1:
            if i[3] == 0:
                continue
            if i[3] == 1:
                valsum += 1 * ATTACK_VALUE
            if i[3] == 2:
                valsum += 2 * ATTACK_VALUE
        if i[1] == 2:
            if i[3] == 0:
                continue
            if i[3] == 1:
                valsum += 3 * ATTACK_VALUE
            if i[3] == 2:
                valsum += 4 * ATTACK_VALUE
        if i[1] == 3:
            if i[3] == 0:
                continue
            if i[3] == 1:
                valsum += 5 * ATTACK_VALUE
            if i[3] == 2:
                valsum += 10 * ATTACK_VALUE
        if i[1] == 4:
            if i[3] == 0:
                continue
            if i[3] == 1:
                valsum += 500 * ATTACK_VALUE
            if i[3] == 2:
                valsum += 1000000 * ATTACK_VALUE

    for i in node.opponent_analysis_copy:
        if i[1] == 1:
            if i[3] == 0:
                continue
            if i[3] == 1:
                valsum += -1 * DEFENSE_VALUE
            if i[3] == 2:
                valsum += -2 * DEFENSE_VALUE
        if i[1] == 2:
            if i[3] == 0:
                continue
            if i[3] == 1:
                valsum += -3 * DEFENSE_VALUE
            if i[3] == 2:
                valsum += -4 * DEFENSE_VALUE
        if i[1] == 3:
            if i[3] == 0:
                continue
            if i[3] == 1:
                valsum += -5 * DEFENSE_VALUE
            if i[3] == 2:
                valsum += -10 * DEFENSE_VALUE
        if i[1] == 4:
            if i[3] == 0:
                continue
            if i[3] == 1:
                valsum += -500 * DEFENSE_VALUE
            if i[3] == 2:
                valsum += -1000000 * DEFENSE_VALUE

    return valsum

#created a new node for every free location on the board
#tried to score that position using the list of connections and the scoring funtion above
def immediate_best(node):
    listofmoves = []

    for x_pos in range(node.BOARD_SIZE):
        for y_pos in range(node.BOARD_SIZE):

            location = (x_pos, y_pos)

            if legalMove(node.board, location):
                new_board = copy.deepcopy(node.board)
                new_board[x_pos][y_pos] = ID
                new_node = Node(new_board, node.BOARD_SIZE, node.player_analysis_copy, node.opponent_analysis_copy,  coordinates=location)
                new_node.our_moves_copy.append(location)
                new_node.last_move_ours_copy = new_node.our_moves_copy[-1]

                check_neighbours(ID, new_node.board, new_node.player_analysis_copy)
                
                check_arr_ours = filter_representation(new_node.player_analysis_copy, 1)
                new_node.player_analysis_copy = check_connection_duplicates(check_arr_ours, player_analysis)


                #print(new_node.coordinates)
                new_node.value = valCalc(new_node)
                listofmoves.append(new_node)

    minimax = 100
    bestNode = node

    #print(listofmoves[0].value)
    for i in range(len(listofmoves)):
        if listofmoves[i].value <= minimax:
            bestNode = listofmoves[i]
            minimax = listofmoves[i].value

    return bestNode
"""
