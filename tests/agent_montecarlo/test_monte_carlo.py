import numpy as np
from agents.common import NO_PLAYER, PLAYER1, PLAYER2, \
    initialize_game_state, PlayerAction, BoardPiece
from tests.test_boards import TestBoards
import tests.test_trees as tt
from agents.agent_montecarlo.monte_carlo import MonteCarlo


def test_init():
    """
    Test __init()__ function of MonteCarlo class by checking if all class
    attributes are correctly initialized.
    """
    from agents.agent_montecarlo.monte_carlo import MonteCarlo
    from agents.agent_montecarlo.monte_carlo_node import MonteCarloNode

    explore_param = np.sqrt(2)

    board = initialize_game_state()
    player = np.random.choice([PLAYER1, PLAYER2])

    ret = MonteCarlo(board, player, explore_param=explore_param)
    assert isinstance(ret, MonteCarlo)
    assert ret.explore_param == explore_param
    assert isinstance(ret.root, MonteCarloNode)

    assert np.all(ret.root.board == board)
    assert ret.root.player == player
    assert ret.root.last_action is None
    assert ret.root.n_simulations == 0
    assert ret.root.n_wins == 0
    assert ret.root.parent is None
    assert len(ret.root.children) == 7
    for key in ret.root.children:
        assert not ret.root.children[key]


def test_select():
    """
    Test function select() of class MonteCarlo by checking the following
    scenarios:
        returns given node if it is not fully expanded
        returns childnode with maximal UCB1 value otherwise

    """
    from agents.agent_montecarlo.monte_carlo_node import MonteCarloNode

    mcst = tt.create_empty_tree()
    ret = mcst.select(mcst.root)
    assert isinstance(ret, MonteCarloNode)
    assert ret == mcst.root

    action = np.random.randint(0, 7)
    mcst.root.expand(action)
    ret = mcst.select(mcst.root)
    assert ret == mcst.root

    mcst.root = tt.create_ucb_node()
    ret = mcst.select(mcst.root)
    ucb = np.ones(7)
    for child in mcst.root.children:
        ucb[child] = mcst.root.children[child].UCB1(explore_param=np.sqrt(2))
    assert ret.last_action == np.argmax(ucb)


def test_simulate():
    """
    Test the function simulate() of class MonteCarlo by checking return type
    and the simulation of two created sample boards from the class
    TestBoards which always have a known outcome.
    """
    mcst = tt.create_empty_tree()
    ret = mcst.simulate(mcst.root)
    assert isinstance(ret, BoardPiece) or ret is None

    mcst.root.board = TestBoards.board_win_player1
    ret = mcst.simulate(mcst.root)
    assert ret == PLAYER1

    mcst.root.board = TestBoards.board_drawn
    ret = mcst.simulate(mcst.root)
    print(mcst.root.board)
    print(mcst.root.player)
    assert ret is None


# phase 4 - Backpropagation: Update ancestor statistics
def test_backprop_small():

    mcst = tt.create_empty_tree()
    mcst.root = tt.create_ucb_node()

    # one Monte Carlo Step
    selected_node = mcst.select(mcst.root)
    action_expand = np.random.choice(selected_node.unexpanded_actions())
    new_child = selected_node.expand(action_expand)
    winner = mcst.simulate(new_child)

    ancestors = [new_child, new_child.parent]
    n_s, n_w = [], []
    for node in ancestors:
        n_s.append(node.n_simulations)
        n_w.append(node.n_wins)

    mcst.backprop(new_child, winner)

    n_s_after, n_w_after = [], []
    for node in ancestors:
        n_s_after.append(node.n_simulations)
        n_w_after.append(node.n_wins)

    assert all(np.array(n_s) + 1 == np.array(n_s_after))
    for i, node in enumerate(ancestors):
        if node.player == winner:
            assert n_w_after[i] == n_w[i]
        else:
            assert n_w_after[i] == n_w[i]+1


def test_backprop_sim():

    # build a bigger example tree
    mcst, new_child, winner = tt.simulate_tree()

    # get current n values:
    ancestors = [new_child,
                 new_child.parent,
                 new_child.parent.parent,
                 new_child.parent.parent.parent,
                 new_child.parent.parent.parent.parent]
    n_s, n_w = [], []
    for node in ancestors:
        n_s.append(node.n_simulations)
        n_w.append(node.n_wins)

    mcst.backprop(new_child, winner)

    n_s_after, n_w_after = [], []
    for node in ancestors:
        n_s_after.append(node.n_simulations)
        n_w_after.append(node.n_wins)

    assert all(np.array(n_s) + 1 == np.array(n_s_after))
    for i, node in enumerate(ancestors):
        if node.player == winner:
            assert n_w_after[i] == n_w[i]
        else:
            assert n_w_after[i] == n_w[i]+1


def test_best_action():
    """
    Tests best_action by checking if it correctly chooses the childnode with
    the most wins (or most simulations respectively).
    """
    mcst, new_child, winner = tt.simulate_tree()
    ret = mcst.best_action()
    assert isinstance(ret, PlayerAction)

    n_wins = np.zeros(7)
    for child in mcst.root.children:
        n_wins[child] = mcst.root.children[child].n_wins

    assert ret == np.argmax(n_wins)

    mcst.best_action(mode='n_sim')
    n_sim = np.zeros(7)
    for child in mcst.root.children:
        n_sim[child] = mcst.root.children[child].n_simulations
    assert ret == np.argmax(n_sim)


def test_run_search():
    mcst = tt.create_empty_tree()
    mcst.run_search(timeout=1)

    #


def test_generate_move():
    """
    Test generate_move function of montecarlo agent of the following:
        - return is Tuple of (PlayerAction, MonteCarlo)
        - returned column is in range of board
        - returned column is not full
        - if new root in search tree is correctly assigned for saved state

    """
    from agents.agent_montecarlo import generate_move

    board = TestBoards.board_test_4
    player = PLAYER2
    action_opponent = 3
    mcst, _, _ = tt.simulate_tree()
    saved_state = {PLAYER1: action_opponent, PLAYER2: mcst}

    old_root = mcst.root
    action, new_mcts = generate_move(board, player, saved_state)
    new_root = old_root.children[action_opponent].children[action]

    assert new_mcts.root == new_root
    assert new_mcts.root.n_simulations == new_root.n_simulations
    assert new_mcts.root.n_wins == new_root.n_wins
    assert np.all(new_mcts.root.board == new_root.board)

    assert isinstance(action, PlayerAction)
    assert isinstance(new_mcts, MonteCarlo)
    assert action in np.arange(0, board.shape[1], 1)
    assert board[board.shape[0]-1, action] == NO_PLAYER


def test_generate_move_win():
    """
    Tests if generate move takes direct win and blocks direct loss in next
    round by using a simple example board of the TestBoards class.
    """
    from agents.agent_montecarlo import generate_move

    # test if takes the win if possible
    player = PLAYER1
    saved_state = {PLAYER1: None, PLAYER2: None}
    action, mcst = generate_move(TestBoards.board_mc1, player, saved_state)
    assert action == 3

    saved_state = {PLAYER1: None, PLAYER2: None}
    player = PLAYER2
    action, mcst = generate_move(TestBoards.board_mc1, player, saved_state)
    assert action == 3


def test_generate_move_direct_win():
    """
    Tests if generate move takes direct win and blocks direct loss in next
    round by using a simple example board of the TestBoards class.
    """
    from agents.agent_montecarlo import generate_move

    # test if takes the win if possible
    player = PLAYER1
    saved_state = {PLAYER1: None, PLAYER2: None}
    action, mcst = generate_move(TestBoards.board_mc1, player, saved_state)
    assert action == 3

    saved_state = {PLAYER1: None, PLAYER2: None}
    player = PLAYER2
    action, mcst = generate_move(TestBoards.board_mc1, player, saved_state)
    assert action == 3









def test_generate_move_sampleboards():
    """
    Test function max_player_move by checking generated moves for the
    following example boards.
    """
    from agents.agent_montecarlo import generate_move


# assert if max agent plays 1 or 4 to prevent win of min agent in second
    # next move
    board = TestBoards.board_minimax_depth2
    print(TestBoards.minimax_depth2)

    player = PLAYER1
    saved_state = {PLAYER1: 5, PLAYER2: None}
    action, mcst = generate_move(board, player, saved_state)
    print(action)
    assert action == 1 or action == 4

    # assert if max agent plays 1 to prevent win of min agent in next move
    board = TestBoards.board_minimax_2
    print(TestBoards.minimax_2)
    saved_state = {PLAYER1: 5, PLAYER2: None}
    action, mcst = generate_move(board, player, saved_state)
    print(action)
    assert action == 1

    # assert if max agent plays 1 to win
    board = TestBoards.board_minimax_2
    player = PLAYER2
    print(TestBoards.minimax_2)
    saved_state = {PLAYER1: 5, PLAYER2: None}
    action, mcst = generate_move(board, player, saved_state)
    print(action)
    assert action == 1
