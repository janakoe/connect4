import numpy as np
import re
#from agents.common import initialize_game_state, BoardPiecePrint, NO_PLAYER, NO_PLAYER_PRINT, PLAYER1, P
from agents.common import *

board = initialize_game_state()
#print(board)
board[0,0] = PLAYER1
board[0,2] = PLAYER2
s = pretty_print_board(board)

pp_board = s
print(pp_board)

top = '|=============|\n'
bottom = '|=============|\n|0 1 2 3 4 5 6|'

pp_board = pp_board[len(top):(len(s)-len(bottom))-1]
pp_board = np.array(pp_board.split('\n'))

print(pp_board)
test = pp_board[5][1:-1:2]
testtest = np.array(test.split())
print(testtest)
print(test, len(test), type(test))

board_print = np.empty((6, 7), dtype= BoardPiecePrint)

for i in range(len(board_print)):
    row = pp_board[i][1:-1:2]
    for j in range(len(board_print[0])):
        board_print[i,j] = row[j]

print(board_print)


board= np.empty_like(board_print, dtype= BoardPiece)
board[board_print == NO_PLAYER_PRINT] = NO_PLAYER
board[board_print == PLAYER1_PRINT] = PLAYER1
board[board_print == PLAYER2_PRINT] = PLAYER2
board = np.flip(board, 0)
print(board)


