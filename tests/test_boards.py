from agents.common import string_to_board, connected_n, PLAYER1, PlayerAction


class TestBoards:
    """
    Generates example boards for the tests
    """

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

    str_3_1 = '|=============|\n' \
              '|             |\n' \
              '|             |\n' \
              '|             |\n' \
              '|X   X O      |\n' \
              '|X   O X O    |\n' \
              '|X O X O O O  |\n' \
              '|=============|\n' \
              '|0 1 2 3 4 5 6|'

    board_3_1 = string_to_board(str_3_1)

    str_3_2 = '|=============|\n' \
              '|             |\n' \
              '|X            |\n' \
              '|O     O     X|\n' \
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
                  '|O   X        |\n' \
                  '|O X X   O X  |\n' \
                  '|X O X O X O  |\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'

    board_heuristic_1 = string_to_board(heuristic_1)

    heuristic_2 = '|=============|\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|            O|\n' \
                  '|O           X|\n' \
                  '|X O   O   X X|\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'

    board_heuristic_2 = string_to_board(heuristic_2)

    evaluate_1 =  '|=============|\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|O       X    |\n' \
                  '|O X X   O X  |\n' \
                  '|X O X O X O  |\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'

    board_evaluate_1 = string_to_board(evaluate_1)

    minimax_depth2 =  '|=============|\n' \
                      '|             |\n' \
                      '|             |\n' \
                      '|             |\n' \
                      '|             |\n' \
                      '|            X|\n' \
                      '|    O O     X|\n' \
                      '|=============|\n' \
                      '|0 1 2 3 4 5 6|'

    board_minimax_depth2 = string_to_board(minimax_depth2)

    minimax_2 = '|=============|\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|             |\n' \
                  '|    O O O X X|\n' \
                  '|X O O O X X X|\n' \
                  '|=============|\n' \
                  '|0 1 2 3 4 5 6|'

    board_minimax_2 = string_to_board(minimax_2)

    minimax_3 = '|=============|\n' \
                '|             |\n' \
                '|             |\n' \
                '|             |\n' \
                '|             |\n' \
                '|      O      |\n' \
                '|      X X    |\n' \
                '|=============|\n' \
                '|0 1 2 3 4 5 6|'

    board_minimax_3 = string_to_board(minimax_3)

    str_drawn = '|=============|\n' \
                '|O X O X      |\n' \
                '|O X O X X O O|\n' \
                '|O X O O X X X|\n' \
                '|X O X X O O O|\n' \
                '|X X O X O X X|\n' \
                '|X O X O O O X|\n' \
                '|=============|\n' \
                '|0 1 2 3 4 5 6|'

    board_drawn = string_to_board(str_drawn)

    str_win_player1 = '|=============|\n' \
                      '|O X X X      |\n' \
                      '|O X O X X X O|\n' \
                      '|O X O O X X X|\n' \
                      '|X O X X O X O|\n' \
                      '|X X O X O O X|\n' \
                      '|X O X O O O X|\n' \
                      '|=============|\n' \
                      '|0 1 2 3 4 5 6|'

    board_win_player1 = string_to_board(str_win_player1)


    mc1 = '|=============|\n' \
                   '|             |\n' \
                   '|             |\n' \
                   '|             |\n' \
                   '|             |\n' \
                   '|             |\n' \
                   '|X X X        |\n' \
                   '|=============|\n' \
                   '|0 1 2 3 4 5 6|'

    board_mc1 = string_to_board(mc1)
