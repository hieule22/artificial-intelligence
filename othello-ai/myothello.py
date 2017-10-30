# Name: Hieu Le - htl5683@truman.edu

# File myothello.py
# Implements the heuristic evaluation function for an Othello-playing AI.

from othello import *


class HieuPlayer(othello_player):
    def initialize(self, boardstate, totalTime, color):
        print("Initializing", self.name)
        self.mycolor = color
        # Each square on the grid is weighted by a relative score. To take
        # advantage of symmetry, only the weight matrix for the upper left
        # quadrant of the board is provided.
        self.weight_matrix = [[120, -20, 20, 5],
                              [-20, -40, -5, -5],
                              [20, -5, 15, 3],
                              [5, -5, 3, 3]]
        pass

    def calculate_utility(self, boardstate):
        # The difference between the number of discs gives a weak evaluation of
        # current board state.
        utility = 1 * self.mycount_difference(boardstate)

        # The number of legal moves in the current board state gives a stronger
        # evaluation of current board state.
        num_next_moves = len(boardstate.legal_moves())
        if boardstate.getPlayer() == self.mycolor:
            utility += 10 * num_next_moves
        else:
            utility -= 10 * num_next_moves

        # Certain positions in the grid are strategic positions that result in
        # great advantage when occupied. For example, a disc in a corner square
        # could never be flipped.
        for row in range(1, 9):
            for col in range(1, 9):
                index = row * 10 + col
                weight = self.get_weight(row, col)
                if boardstate._board[index] == self.mycolor:
                    utility += weight
                elif boardstate._board[index] != Empty:
                    utility -= weight

        return utility

    def get_weight(self, row, col):
        """Return the relative weight for a given square on the grid"""
        if row > 4:
            row = 9 - row
        if col > 4:
            col = 9 - col
        return self.weight_matrix[row - 1][col - 1]

    def alphabeta_parameters(self, boardstate, remainingTime):
        return 4, None, None

    def mycount_difference(self, boardstate):
        return (boardstate._board.count(self.mycolor) -
                boardstate._board.count(opponent(self.mycolor)))


start_graphical_othello_game(othello_player("Foo"), HieuPlayer("Hieu"))
