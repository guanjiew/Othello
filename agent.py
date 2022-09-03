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


def get_opp_color(color):
    if color == 1:
        opponent = 2
    else:
        opponent = 1
    return opponent


# Heuristic Description:
# The Heuristic is a weighted sum of three parts:
# the coin difference (compute_utility),
# the choice difference (compute_choice),
# the corner difference (compute_corner)

# The coin difference indicates the difference between the color of our coin and the opponent's coin
# Although this is the utility value of terminal state, it is not an accurate estimate during the game, hence give it
# a small weight 0.2 in our case

# The choice difference indicates the difference between the number of possible moves of our coin and the number of
# possible moves of the opponent's coin This is also an important estimate since the number of moves one can make
# usually indicate the mobility, hence give it a small weight 0.2 in our case

# The number of occupied corners between our agent and opponent is the most important estimate, since corner
# positions are the most important locations in the board. This is because they are unchangeable, and tends to be a
# good place to s start to flip the opponent's coin, hence give it a large weight 0.6 in our case

# Method to compute utility value of each state
def compute_utility(board, color):
    if color == 1:
        utility = get_score(board)[0] - get_score(board)[1]
    else:
        utility = get_score(board)[1] - get_score(board)[0]
    return utility


# Method to compute choice value of each states
def compute_choice(board, color):
    opponent = get_opp_color(color)
    max_choice = len(get_possible_moves(board, color))
    min_choice = len(get_possible_moves(board, opponent))
    if max_choice == 0 and min_choice == 0:
        return 0
    else:
        return max_choice - min_choice


# Method to compute corner value of each state
def compute_corner(board, color):
    opponent = get_opp_color(color)
    corner_pos = [(0, 0), (0, len(board) - 1), (len(board) - 1, 0), (len(board) - 1, len(board) - 1)]
    max_corner = 0
    min_corner = 0
    for corner in corner_pos:
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
    min_color = get_opp_color(color)
    possible_moves = get_possible_moves(board, min_color)
    if possible_moves == [] or limit == 0:
        if caching:
            if board not in caching_states:
                caching_states[board] = compute_utility(board, color)
            return best_move, caching_states[board]
        else:
            return best_move, compute_utility(board, color)
    for move in possible_moves:
        state = play_move(board, min_color, move[0], move[1])
        _, nxt_value = minimax_max_node(state, color, limit - 1, caching)
        if nxt_value < value:
            best_move = move
            value = nxt_value
    return best_move, value


def minimax_max_node(board, color, limit, caching=0):
    value = float("-inf")
    best_move = None
    possible_moves = get_possible_moves(board, color)
    if possible_moves == [] or limit == 0:
        if caching:
            if board not in caching_states:
                caching_states[board] = compute_utility(board, color)
            return best_move, caching_states[board]
        else:
            return best_move, compute_utility(board, color)

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
    caching_states.clear()
    move, _ = minimax_max_node(board, color, limit, caching)
    caching_states.clear()
    return move


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    value = float("inf")
    best_move = None
    min_color = get_opp_color(color)
    possible_moves = get_possible_moves(board, min_color)
    if possible_moves == [] or limit == 0:
        if caching:
            if board not in caching_states:
                caching_states[board] = compute_utility(board, color)
            return best_move, caching_states[board]
        else:
            return best_move, compute_utility(board, color)
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
    value = float("-inf")
    best_move = None
    possible_moves = get_possible_moves(board, color)
    if possible_moves == [] or limit == 0:
        if caching:
            if board not in caching_states:
                caching_states[board] = compute_utility(board, color)
            return best_move, caching_states[board]
        else:
            return best_move, compute_utility(board, color)
    if ordering:
        possible_moves = sorted(possible_moves,
                                key=lambda moves: compute_utility(play_move(board, color, moves[0], moves[1]), color),
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
    caching_states.clear()
    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI")  # First line is the name of this AI
    arguments = input().split(",")

    color = int(arguments[0])  # Player color: 1 for dark (goes first), 2 for light.
    limit = int(arguments[1])  # Depth limit
    minimax = int(arguments[2])  # Minimax or alpha beta
    caching = int(arguments[3])  # Caching
    ordering = int(arguments[4])  # Node-ordering (for alpha-beta only)

    if (minimax == 1):
        eprint("Running MINIMAX")
    else:
        eprint("Running ALPHA-BETA")

    if (caching == 1):
        eprint("State Caching is ON")
    else:
        eprint("State Caching is OFF")

    if (ordering == 1):
        eprint("Node Ordering is ON")
    else:
        eprint("Node Ordering is OFF")

    if (limit == -1):
        eprint("Depth Limit is OFF")
    else:
        eprint("Depth Limit is ", limit)

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

            # Select the move and send it to the manager
            if (minimax == 1):  # run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else:  # else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)

            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
