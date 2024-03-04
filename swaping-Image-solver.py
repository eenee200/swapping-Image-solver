from queue import Queue
import heapq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import copy
import time
from queue import PriorityQueue
import math

class PuzzlePiece:
    def __init__(self, image_patch, position):
        self.image_patch = image_patch
        self.position = position

    def __lt__(self, other):
        # Define a comparison method to compare PuzzlePiece objects
        return self.position < other.position  # You can change the comparison criterion as needed

    def __eq__(self, other):
        # Define an equality method to compare PuzzlePiece objects
        return self.position == other.position  # You can change the comparison criterion as needed

    def __hash__(self):
        # Define a hash method to make PuzzlePiece objects hashable
        return hash(self.position)


class ImagePuzzle:
    def __init__(self, image_path, rows, cols):
        self.image = Image.open(image_path)
        self.rows = rows
        self.cols = cols
        self.piece_width = self.image.width // cols
        self.piece_height = self.image.height // rows
        self.pieces = self.split_image()

    def split_image(self):
        pieces = []
        for i in range(self.rows):
            for j in range(self.cols):
                x = j * self.piece_width
                y = i * self.piece_height
                patch = self.image.crop((x, y, x + self.piece_width, y + self.piece_height))
                pieces.append(PuzzlePiece(patch, (i, j)))
        return pieces

    def shuffle(self):
        shuffled_pieces = copy.deepcopy(self.pieces)
        random.shuffle(shuffled_pieces)
        return shuffled_pieces

    def show(self, pieces=None):
        if pieces is None:
            pieces = self.pieces
        new_image = np.ones((self.image.height, self.image.width, 3), dtype=np.uint8) * 255  # Create a blank image with white background
        for i, piece in enumerate(pieces):
            x = (i % self.cols) * self.piece_width
            y = (i // self.cols) * self.piece_height
            new_image[y:y+self.piece_height, x:x+self.piece_width] = np.array(piece.image_patch)
        plt.imshow(new_image)
        plt.axis('off')
        plt.show()

    

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __lt__(self, other):
        return self.path_cost < other.path_cost

def is_goal_state(state, goal_state):
    return state == goal_state
def get_possible_moves(state):
    """
    Generate possible moves from the current state.
    """
    moves = []
    rows = len(state)
    cols = len(state[0])

    for i in range(rows):
        for j in range(cols):
            # Generate moves for swapping neighboring pieces
            if i > 0:  # Check if not at top row
                new_state = copy.deepcopy(state)
                new_state[i][j], new_state[i - 1][j] = new_state[i - 1][j], new_state[i][j]
                moves.append(new_state)
            if j > 0:  # Check if not at leftmost column
                new_state = copy.deepcopy(state)
                new_state[i][j], new_state[i][j - 1] = new_state[i][j - 1], new_state[i][j]
                moves.append(new_state)
            if i < rows - 1:  # Check if not at bottom row
                new_state = copy.deepcopy(state)
                new_state[i][j], new_state[i + 1][j] = new_state[i + 1][j], new_state[i][j]
                moves.append(new_state)
            if j < cols - 1:  # Check if not at rightmost column
                new_state = copy.deepcopy(state)
                new_state[i][j], new_state[i][j + 1] = new_state[i][j + 1], new_state[i][j]
                moves.append(new_state)

    return moves
class AStarSearch:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.execution_time = None

    def h(self, state):
        # Heuristic function: Manhattan distance
        total_distance = 0
        for i in range(len(state)):
            for j in range(len(state[0])):
                piece = state[i][j]
                goal_position = self.goal_state[piece.position[0]][piece.position[1]].position
                total_distance += abs(i - goal_position[0]) + abs(j - goal_position[1])
        return total_distance

    def g(self, state):
        # Cost function: path cost from the initial state
        return 1

    def f(self, state):
        # Evaluation function: f(n) = g(n) + h(n)
        return self.g(state) + self.h(state)

    def solve(self):
        start_time = time.time()  # Record start time
        open_set = PriorityQueue()
        open_set.put((self.f(self.initial_state), self.initial_state))
        came_from = {}
        g_score = {tuple(map(tuple, self.initial_state)): 0}

        while not open_set.empty():
            current_state = open_set.get()[1]

            if is_goal_state(current_state, self.goal_state):
                self.execution_time = time.time() - start_time  # Calculate execution time
                return self.reconstruct_path(current_state, came_from)

            for next_state in get_possible_moves(current_state):
                tentative_g_score = g_score[tuple(map(tuple, current_state))] + self.g(next_state)
                if tuple(map(tuple, next_state)) not in g_score or tentative_g_score < g_score[tuple(map(tuple, next_state))]:
                    came_from[tuple(map(tuple, next_state))] = current_state
                    g_score[tuple(map(tuple, next_state))] = tentative_g_score
                    open_set.put((self.f(next_state), next_state))

        self.execution_time = time.time() - start_time  # Calculate execution time
        return None
    def solve_with_alpha_beta(self):
        start_time = time.time()  # Record start time
        open_set = PriorityQueue()
        open_set.put((self.f(self.initial_state), self.initial_state))
        came_from = {}
        g_score = {tuple(map(tuple, self.initial_state)): 0}
        alpha = float("-inf")
        beta = float("inf")

        while not open_set.empty():
            current_state = open_set.get()[1]

            if is_goal_state(current_state, self.goal_state):
                self.execution_time = time.time() - start_time  # Calculate execution time
                return self.reconstruct_path(current_state, came_from)

            for next_state in get_possible_moves(current_state):
                tentative_g_score = g_score[tuple(map(tuple, current_state))] + self.g(next_state)
                if tuple(map(tuple, next_state)) not in g_score or tentative_g_score < g_score[tuple(map(tuple, next_state))]:
                    came_from[tuple(map(tuple, next_state))] = current_state
                    g_score[tuple(map(tuple, next_state))] = tentative_g_score
                    f_score = self.f(next_state)
                    open_set.put((f_score, next_state))

                    # Alpha-Beta pruning
                    if f_score >= beta:
                        break
                    alpha = max(alpha, f_score)

        self.execution_time = time.time() - start_time  # Calculate execution time
        return None
    def reconstruct_path(self, current_state, came_from):
        total_path = [current_state]
        while tuple(map(tuple, current_state)) in came_from:
            current_state = came_from[tuple(map(tuple, current_state))]
            total_path.insert(0, current_state)
        return total_path


# Load the image
image_path = 'pic.jpg'  # Path to your image
rows = 3
cols = 3

puzzle = ImagePuzzle(image_path, rows, cols)
shuffled_pieces = puzzle.shuffle()
puzzle.show()
puzzle.pieces = shuffled_pieces
puzzle.show()

initial_state = [[shuffled_pieces[i * cols + j] for j in range(cols)] for i in range(rows)]

# Define the goal state with PuzzlePiece objects
goal_state = [[PuzzlePiece(None, (i, j)) for j in range(cols)] for i in range(rows)]

# Print the goal state with positions
print("Goal State:")
for row in goal_state:
    print(" ".join(str(piece.position) for piece in row))
print("initial_state:")
for row in initial_state:
    print(" ".join(str(piece.position) for piece in row))

a_star = AStarSearch(initial_state, goal_state)
solution = a_star.solve()

if solution:
    print("Solution found!")
    for step, state in enumerate(solution):
        print("Step", step + 1)
        for row in state:
            print(" ".join(str(piece.position) for piece in row))
        print()
    final_state = solution[-1]
    flat_final_state = [piece for row in final_state for piece in row]
    puzzle.pieces = flat_final_state  # Deep copy the pieces
    
    # Display the solved puzzle
    puzzle.show()
    
    # Display the composite image
    print("Solution found in {:.6f} seconds!".format(a_star.execution_time))
else:
    print("No solution found.")


# Solve the puzz
solution = a_star.solve_with_alpha_beta()

if solution:
    print("Solution found!")
    for step, state in enumerate(solution):
        print("Step", step + 1)
        for row in state:
            print(" ".join(str(piece.position) for piece in row))
        print()
    final_state = solution[-1]
    flat_final_state = [piece for row in final_state for piece in row]
    puzzle.pieces = flat_final_state  # Deep copy the pieces
    
    # Display the solved puzzle
    puzzle.show()
    
    # Display the composite image
    print("Solution found in {:.6f} seconds!".format(a_star.execution_time))
else:
    print("No solution found.")
