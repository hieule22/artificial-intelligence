# File eightPuzzle.py
# Implements the Eight Puzzle problem for state space search

#                Node Expansions
# Problem | BFS   | A*(tiles) | A*(dist)
#    A      7          3          3
#    B      69         8          7
#    C      183        23         10
#    D      807        48         30
#    E      831        48         29
#    F      1557       100        21
#    G      6005       331        42
#    H      46690      3233       198


from informedSearch import *


class EightPuzzleState(InformedProblemState):
    """
    Each state in the Eight Puzzle problem is characterized by a 2-dimensional array representing the current grid.
    The grid is filled with numbers while the blank square is denoted with a 0 value.
    """

    def __init__(self, grid):
        self.grid = grid
        if grid is not None:
            # Extract the position of the blank square from the current grid.
            for row in range(len(grid)):
                for col in range(len(grid[row])):
                    if grid[row][col] == 0:
                        self.blank_row, self.blank_col = row, col
                        return

    def __str__(self):
        """Returns a string representation of this state"""
        rep = ""
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                rep += '%s ' % self.grid[row][col]
            rep += '\n'
        return rep

    def illegal(self):
        """Tests whether this state is illegal"""
        return self.grid is None

    def equals(self, state):
        """Determines whether the state instance and the given state are equal"""
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if not self.grid[row][col] == state.grid[row][col]:
                    return False
        return True

    # Each operator corresponds to shifting the blank square in each of the four possible directions.
    # This induces changes in the row and column number of the blank square.
    OPERATORS = [[-1, 0], [0, 1], [1, 0], [0, -1]]

    def operatorNames(self):
        """
        Returns a list of operator names in the same order as the applyOperators method.
        The returned name corresponds to the action applied on the numbered square that is moved into the blank square.
        """
        return ['Slide Down', 'Slide Right', 'Slide Up', 'Slide Left']

    def applyOperators(self):
        """Returns a list of possible successors to the current state, some of which maybe illegal."""
        next_states = []

        for operator in self.OPERATORS:
            next_board = [row[:] for row in self.grid]
            # Compute the new position of the blank square.
            next_blank_row, next_blank_col = self.blank_row + operator[0], self.blank_col + operator[1]

            if 0 <= next_blank_row < len(next_board) and 0 <= next_blank_col < len(next_board[next_blank_row]):
                next_board[self.blank_row][self.blank_col], next_board[next_blank_row][next_blank_col] = \
                    next_board[next_blank_row][next_blank_col], next_board[self.blank_row][self.blank_col]
                next_states.append(EightPuzzleState(next_board))
            else:
                next_states.append(EightPuzzleState(None))

        return next_states

    def heuristic(self, goal):
        # return 0
        # return self.heuristic_hamming(goal)
        return self.heuristic_manhattan(goal)

    def heuristic_hamming(self, other):
        hamming_distance = 0
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if not self.grid[row][col] == other.grid[row][col]:
                    hamming_distance += 1
        return hamming_distance

    def heuristic_manhattan(self, other):
        manhattan_distance = 0
        for expected_row in range(len(other.grid)):
            for expected_col in range(len(other.grid[expected_row])):
                for actual_row in range(len(self.grid)):
                    for actual_col in range(len(self.grid[actual_row])):
                        if other.grid[expected_row][expected_col] == self.grid[actual_row][actual_col]:
                            manhattan_distance += abs(expected_row - actual_row) + abs(expected_col - actual_col)
        return manhattan_distance


goalState = EightPuzzleState(
    [[1, 2, 3],
     [8, 0, 4],
     [7, 6, 5]])

initialStateBoards = [
    [[1, 3, 0],
     [8, 2, 4],
     [7, 6, 5]],

    [[1, 3, 4],
     [8, 6, 2],
     [0, 7, 5]],

    [[0, 1, 3],
     [4, 2, 5],
     [8, 7, 6]],

    [[7, 1, 2],
     [8, 0, 3],
     [6, 5, 4]],

    [[8, 1, 2],
     [7, 0, 4],
     [6, 5, 3]],

    [[2, 6, 3],
     [4, 0, 5],
     [1, 8, 7]],

    [[7, 3, 4],
     [6, 1, 5],
     [8, 0, 2]],

    [[7, 4, 5],
     [6, 0, 3],
     [8, 1, 2]]
]

InformedSearch(EightPuzzleState(initialStateBoards[5]), goalState)
