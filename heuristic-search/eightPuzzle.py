# Name: Hieu Le - htl5683@truman.edu
# Name: Anh Nguyen - adn6627@truman.edu

# File eightPuzzle.py
# Implements the Eight Puzzle problem for state space search

#                Node Expansions
# Problem | BFS   | A*(tiles) | A*(dist) | Steps
#    A      7          3          3         2
#    B      69         8          7         6
#    C      183        18         9         8
#    D      807        40         24        10
#    E      831        40         24        10
#    F      1557       95         18        12
#    G      6005       269        46        15
#    H      46690      3616       183       20


from informedSearch import *


class EightPuzzleState(InformedProblemState):
    """
    Each state in the Eight Puzzle problem is characterized by a 2-dimensional
    array representing the puzzle grid. The grid is filled with numbers with
    the blank square denoted with a 0 value.
    """

    def __init__(self, grid):
        self.grid = grid
        if grid is not None:
            # Extract the position of the blank square from the current grid.
            self.blank_row, self.blank_col = self.find_position(0)

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
        """
        Determines whether the state instance and the given state are equal
        """
        return self.grid == state.grid

    # Each operator corresponds to shifting the blank square in each of the four
    # possible directions. This induces changes in the row and column number of
    # the blank square.
    OPERATORS = [[-1, 0], [0, 1], [1, 0], [0, -1]]

    def operatorNames(self):
        """
        Returns a list of operator names in the same order as the applyOperators
        method. The returned name corresponds to the action applied on the
        numbered square that is moved into the blank square.
        """
        return ['Slide Down', 'Slide Left', 'Slide Up', 'Slide Right']

    def applyOperators(self):
        """
        Returns a list of possible successors to the current state, some of
        which maybe illegal.
        """
        next_states = []

        for operator in self.OPERATORS:
            next_board = [row[:] for row in self.grid]
            # Compute the new position of the blank square.
            next_blank_row = self.blank_row + operator[0]
            next_blank_col = self.blank_col + operator[1]

            if (0 <= next_blank_row < len(next_board)
                and 0 <= next_blank_col < len(next_board[next_blank_row])):
                # Exchange two adjacent squares.
                next_board[self.blank_row][self.blank_col], \
                next_board[next_blank_row][next_blank_col] = \
                    next_board[next_blank_row][next_blank_col], \
                    next_board[self.blank_row][self.blank_col]
                next_states.append(EightPuzzleState(next_board))
            else:
                next_states.append(EightPuzzleState(None))

        return next_states

    def heuristic(self, goal):
        """Returns the estimated cost of reaching the goal from this state."""
        # return 0
        # return self.get_hamming_distance(goal)
        return self.get_manhattan_distance(goal)

    def get_hamming_distance(self, other):
        """
        Computes the Hamming distance from this state to other, i.e. the number
        of out-of-place squares
        """
        hamming_distance = 0
        for value in range(1, 9):
            actual_row, actual_col = self.find_position(value)
            expected_row, expected_col = other.find_position(value)
            if not (actual_row == expected_row and actual_col == expected_col):
                hamming_distance += 1
        return hamming_distance

    def get_manhattan_distance(self, other):
        """Computes the Manhattan distance from this state to other"""
        manhattan_distance = 0

        for value in range(1, 9):
            actual_row, actual_col = self.find_position(value)
            expected_row, expected_col = other.find_position(value)
            manhattan_distance += abs(expected_row - actual_row)
            manhattan_distance += abs(expected_col - actual_col)

        return manhattan_distance

    def find_position(self, value):
        """Returns row and column number of the cell containing value"""
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.grid[row][col] == value:
                    return row, col
        return -1, -1


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

InformedSearch(EightPuzzleState(initialStateBoards[0]), goalState)
