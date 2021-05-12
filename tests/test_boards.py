from agents.common import string_to_board

class TestBoards:
    # test connected_four
    # boards == True:

    # boards == False
    str_4_1 = '|=============|\n' \
                  '|X            |\n' \
                  '|X            |\n' \
                  '|O   X O X   X|\n' \
                  '|X   O O O   O|\n' \
                  '|X X O X O   X|\n' \
                  '|X O X O O O X|\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'

    board_test_4 = string_to_board(str_4_1)

    # test connected_n : n=3
    # boards == True
    # last_action = 1, player=X
    str_3_diag = '|=============|\n' \
                 '|             |\n' \
                 '|X            |\n' \
                 '|O           X|\n' \
                 '|X   X       O|\n' \
                 '|X X O X O   X|\n' \
                 '|X O X O O O X|\n' \
                 '|=============|\n' \
                 '|0 1 2 3 4 5 6|'

    board_3_diag = string_to_board(str_3_diag)
    # true: last_action = 0, player = X
    # true: last_action = 3, player = 0
    # true: last_action = 5, player = 0

    str_3_1 = '|=============|\n' \
              '|             |\n' \
              '|             |\n' \
              '|             |\n' \
              '|    X O      |\n' \
              '|X   O X O    |\n' \
              '|X O X O O O  |\n' \
              '|=============|\n' \
              '|0 1 2 3 4 5 6|'

    board_3_1 = string_to_board(str_3_1)

    # boards = False

    # false: last_action = 5, player = O
    # false: last_action = 1, player = X
    # true: last_action = 2, player = X

    str_3_2 = '|=============|\n' \
              '|             |\n' \
              '|X            |\n' \
              '|O     O      X|\n' \
              '|X   X X     O|\n' \
              '|X X O X O   X|\n' \
              '|X O X O O O X|\n' \
              '|=============|\n' \
              '|0 1 2 3 4 5 6|'

    board_3_2 = string_to_board(str_3_2)

    str_valid_move =  '|=============|\n' \
                      '|X X X X O   O|\n' \
                      '|             |\n' \
                      '|             |\n' \
                      '|             |\n' \
                      '|             |\n' \
                      '|             |\n' \
                      '|=============|\n' \
                      '|0 1 2 3 4 5 6|'

    board_valid_move = string_to_board(str_valid_move)

    heuristic_1 = '|=============|\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|O            |\n' \
                  '|X            |\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'

    board_heuristic_1 = string_to_board(heuristic_1)

    heuristic_2 = '|=============|\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|X     O      |\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'

    board_heuristic_2 = string_to_board(heuristic_2)

    heuristic_3 = '|=============|\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|      O      |\n' \
                  '|    X X      |\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'

    board_heuristic_3 = string_to_board(heuristic_3)

    heuristic_4 = '|=============|\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|      X      |\n' \
                  '|X     O      |\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'

    board_heuristic_4 = string_to_board(heuristic_4)
