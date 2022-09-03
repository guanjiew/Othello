import random
import sys
import time

# You can use the functions in othello_shared to write your AI for competition
from othello_shared import find_lines, get_possible_moves, get_score, play_move

# # If you choose to try MCTS, you can make use of the code below
# class MCTS_state():
#     """
#             This sample code gives you a idea of how to store records for each node
#             in the tree. However, you are welcome to modify this part or define your own
#             class.
#     """
#     def __init__(self, ID, parent, child, reward, total, board):
#         self.ID = ID
#         self.parent = parent    # a list of states
#         self.child = child      # a list of states
#         self.reward = reward    # number of win
#         self.total = total      # number of simulation for self and (grand*)children
#         self.board = board
#         self.visited = 0        # 0 -> not visited yet, 1 -> already visited
#
#
# def select_move_MCTS(board, color, limit):
#     """
#                You can add additional help functions as long as this function will return a position tuple
#     """
#     initial_state = MCTS_state(0, [], [], 0, 0, board) # this is just an example. delete it when you start to code.
#     pass

"""
An AI player for Othello.
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

caching_states = {}


def eprint(*args, **kwargs):  # you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)


# Method to compute utility value of terminal state
def compute_utility(board, color):
    if color == 1:
        utility = get_score(board)[0] - get_score(board)[1]
    else:
        utility = get_score(board)[1] - get_score(board)[0]
    return utility


def compute_choice(board, color):
    if color == 0:
        opponent = 1
    else:
        opponent = 0
    max_choice = len(get_possible_moves(board, color))
    min_choice = len(get_possible_moves(board, opponent))
    if max_choice == 0 and min_choice == 0:
        return 0
    else:
        return max_choice - min_choice


def compute_corner(board, color):
    if color == 0:
        opponent = 1
    else:
        opponent = 0
    corners = [(0, 0), (0, len(board) - 1), (len(board) - 1, 0), (len(board) - 1, len(board) - 1)]
    max_corner = 0
    min_corner = 0
    for corner in corners:
        i, j = corner
        if board[i][j] == color:
            max_corner += 1
        elif board[i][j] == opponent:
            min_corner += 1
    if min_corner == 0 and max_corner == 0:
        return 0
    else:
        return max_corner - min_corner


# Better heuristic value of board
def compute_heuristic(board, color):
    return 0.2 * compute_utility(board, color) + 0.2 * compute_choice(board, color) + 0.7 * compute_corner(board, color)


############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching=0):
    # In the minimax_min_node function,
    # you want to choose the move from the opponent's possible moves that has the minimum utility for the max player.
    # Also somewhere in minimax_min_node, you are probably calling minimax_max_node for the max player.
    # So the easiest way to implement this (that passes the sanity checks) is to maintain the max player
    # as the "color" in both minimax_min_node and minimax_max_node,
    # while still making sure to get the moves of the opponent in the min_node.
    value = float("inf")
    best_move = None
    if color == 1:
        max_color = 2
    else:
        max_color = 1
    possible_moves = get_possible_moves(board, max_color)
    if possible_moves == [] or limit == 0:
        if caching:
            if board not in caching_states:
                caching_states[board] = compute_heuristic(board, color)
            return best_move, caching_states[board]
        else:
            return best_move, compute_heuristic(board, color)
    for move in possible_moves:
        state = play_move(board, max_color, move[0], move[1])
        _, nxt_value = minimax_max_node(state, color, limit - 1, caching)
        if nxt_value < value:
            best_move = move
            value = nxt_value
    return best_move, value


def minimax_max_node(board, color, limit, caching=0):  # returns highest possible utility
    value = float("-inf")
    best_move = None
    possible_moves = get_possible_moves(board, color)
    # if color == 1:
    #     min_color = 2
    # else:
    #     min_color = 1
    if possible_moves == [] or limit == 0:
        if caching:
            if board not in caching_states:
                caching_states[board] = compute_heuristic(board, color)
            return best_move, caching_states[board]
        else:
            return best_move, compute_heuristic(board, color)

    for move in possible_moves:
        state = play_move(board, color, move[0], move[1])
        _, nxt_value = minimax_min_node(state, color, limit - 1, caching)
        if nxt_value > value:
            best_move = move
            value = nxt_value
    return best_move, value


def select_move_minimax(board, color, limit, caching=0):
    """
    Given a board and a player color, decide on a move.
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.
    """
    move, _ = minimax_max_node(board, color, limit, caching)
    global caching_states
    caching_states = {}
    # print("moves")
    # print(move)
    return move


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    value = float("inf")
    best_move = None
    if color == 1:
        min_color = 2
    else:
        min_color = 1
    possible_moves = get_possible_moves(board, min_color)
    if possible_moves == [] or limit == 0:
        if caching:
            if board not in caching_states:
                caching_states[board] = -1 * compute_heuristic(board, min_color)
            return best_move, caching_states[board]
        else:
            return best_move, -1 * compute_heuristic(board, min_color)
    for move in possible_moves:
        state = play_move(board, min_color, move[0], move[1])
        nxt_move, nxt_value = alphabeta_max_node(state, color, alpha, beta, limit - 1, caching, ordering)
        if value > nxt_value:
            value, best_move = nxt_value, move
        if value <= alpha:
            return best_move, value
        beta = min(beta, value)

    return best_move, value


def alphabeta_max_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    # IMPLEMENT (and replace the line below)
    value = float("-inf")
    best_move = None
    # min_color = 2 if color == 1 else 1
    possible_moves = get_possible_moves(board, color)
    if possible_moves == [] or limit == 0:
        if caching:
            if board not in caching_states:
                caching_states[board] = compute_heuristic(board, color)
            return best_move, caching_states[board]
        else:
            return best_move, compute_heuristic(board, color)
    if ordering:
        possible_moves = sorted(possible_moves,
                                key=lambda moves: compute_heuristic(play_move(board, color, moves[0], moves[1]), color),
                                reverse=True)
    for move in possible_moves:
        state = play_move(board, color, move[0], move[1])
        nxt_move, nxt_value = alphabeta_min_node(state, color, alpha, beta, limit - 1, caching, ordering)
        if value < nxt_value:
            value, best_move = nxt_value, move
        if value >= beta:
            return best_move, value
        alpha = max(alpha, value)

    return best_move, value


def select_move_alphabeta(board, color, limit, caching=0, ordering=0):
    """
    Given a board and a player color, decide on a move.
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations.
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations.
    """
    alpha = float("-inf")
    beta = float("inf")
    caching_states.clear()
    move, _ = alphabeta_max_node(board, color, alpha, beta, limit, caching, ordering)
    return move


def run_ai():
    """
        Please do not modify this part.
        """
    print("Othello AI")  # First line is the name of this AI
    arguments = input().split(",")

    color = int(arguments[0])  # Player color: 1 for dark (goes first), 2 for light.
    limit = int(arguments[1])  # Iteration limit
    minimax = int(arguments[2])  # not used here
    caching = int(arguments[3])  # not used here
    ordering = int(arguments[4])  # not used here

    if (limit == -1):
        eprint("Iteration Limit is OFF")
    else:
        eprint("Iteration Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True:  # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL":  # Game is over.
            print
        else:
            board = eval(input())  # Read in the input and turn it into a Python
            # object. The format is a list of rows. The
            # squares in each row are represented by
            # 0 : empty square
            # 1 : dark disk (player 1)
            # 2 : light disk (player 2)

            # Uncomment the line below if you choose to use MCTS
            # movei, movej = select_move_MCTS(board, color, limit)

            # Otherwise, use whatever formulation you like! e.g.:
            # movei, movej = select_move_minimax(board, color, limit, caching)
            movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)

            print("{} {}".format(movei, movej))
