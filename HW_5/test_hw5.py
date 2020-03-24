import numpy as np

from HW_5 import is_projective, perform_shift, perform_arc, tree_to_actions, action_to_tree


def test_is_projective_phrase():
    proj_toks = [(1, '', '', 0, ''),
                 (2, '', '', 1, ''),
                 (3, '', '', 5, ''),
                 (4, '', '', 5, ''),
                 (5, '', '', 1, '')]

    assert is_projective(proj_toks) is True


def test_not_projective_phrase():
    proj_toks = [(1, '', '', 2, ''),
                 (2, '', '', 0, ''),
                 (3, '', '', 4, ''),
                 (4, '', '', 2, ''),
                 (5, '', '', 2, ''),
                 (6, '', '', 9, ''),
                 (7, '', '', 9, ''),
                 (8, '', '', 4, '')]
    assert is_projective(proj_toks) is False


def test_is_projective_tree():
    """
    Sanity check for the function is_projective()
    """
    # "From the AP comes this story:" should be projective
    proj_toks = [(1, 'From', 'IN', 3, 'case'),
                 (2, 'the', 'DT', 3, 'det'),
                 (3, 'AP', 'NNP', 4, 'obl'),
                 (4, 'comes', 'VBZ', 0, 'root'),
                 (5, 'this', 'DT', 6, 'det'),
                 (6, 'story', 'NN', 4, 'nsubj'),
                 (7, ':', ':', 4, 'punct')]
    assert is_projective(proj_toks) == True

    # "I saw a man today who is tall" should not be projective
    non_proj_toks = [(1, 'I', 'PRP', 2, 'nsubj'),
                     (2, 'saw', 'VBD', 0, 'root'),
                     (3, 'a', 'DT', 4, 'det'),
                     (4, "man", 'NN', 2, 'obj'),
                     (5, 'today', 'NN', 2, 'nmod'),
                     (6, 'who', 'WP', 8, 'nsubj'),
                     (7, 'is', 'VBZ', 8, 'cop'),
                     (8, 'tall', 'JJ', 4, 'acl:relcl')]
    assert is_projective(non_proj_toks) == False


def test_perform_shift():
    """
    Sanity check for the function perform_shift()
    """
    # Before perform SHIFT
    wbuffer = [3, 2, 1]
    stack = [0]
    arcs = []
    configurations = []
    gold_transitions = []

    # Perform SHIFT
    perform_shift(wbuffer, stack, arcs, configurations, gold_transitions)

    # After perform SHIFT
    assert wbuffer == [3, 2], "The result for wbuffer is not correct"
    assert stack == [0, 1], "The result for stack is not correct"
    assert arcs == [], "The result for arcs is not correct"
    assert configurations == [([3, 2, 1], [0], [])], "The result for configurations is not correct"
    assert gold_transitions == ['SHIFT'], "The result for gold_transitions is not correct"


def test_perform_arc_right():
    # Before perform ARC
    direction = 'RIGHT'
    dep_label = 'punct'
    wbuffer = [5, 4, 3]
    stack = [0, 1, 2]
    arcs = []
    configurations = [([5, 4, 3, 2, 1], [0], []),
                      ([5, 4, 3, 2], [0, 1], [])]
    gold_transitions = ['SHIFT', 'SHIFT']

    # Perform ARC
    perform_arc(direction, dep_label, wbuffer, stack, arcs, configurations, gold_transitions)

    # After perform ARC
    assert wbuffer == [5, 4, 3], "The result for wbuffer is not correct"
    assert stack == [0, 1], "The result for stack is not correct"
    assert arcs == [('punct', 1, 2)], "The result for arcs is not correct"
    assert configurations == [([5, 4, 3, 2, 1], [0], []),
                              ([5, 4, 3, 2], [0, 1], []),
                              ([5, 4, 3], [0, 1, 2], [])], \
        "The result for configurations is not correct"
    assert gold_transitions == ['SHIFT', 'SHIFT', 'RIGHTARC_punct'], "The result for gold_transitions is not correct"


def test_perform_arc_left():
    # Before perform ARC
    direction = 'LEFT'
    dep_label = 'punct'
    wbuffer = [5, 4, 3]
    stack = [0, 1, 2]
    arcs = []
    configurations = [([5, 4, 3, 2, 1], [0], []),
                      ([5, 4, 3, 2], [0, 1], [])]
    gold_transitions = ['SHIFT', 'SHIFT']

    # Perform ARC
    perform_arc(direction, dep_label, wbuffer, stack, arcs, configurations, gold_transitions)

    # After perform ARC
    assert wbuffer == [5, 4, 3], "The result for wbuffer is not correct"
    assert stack == [0, 2], "The result for stack is not correct"
    assert arcs == [('punct', 2, 1)], "The result for arcs is not correct"
    assert configurations == [([5, 4, 3, 2, 1], [0], []),
                              ([5, 4, 3, 2], [0, 1], []),
                              ([5, 4, 3], [0, 1, 2], [])], \
        "The result for configurations is not correct"
    assert gold_transitions == ['SHIFT', 'SHIFT', 'LEFTARC_punct'], "The result for gold_transitions is not correct"


def test_tree_to_actions_b():
    """
    Sanity check for the function tree_to_actions()
    """
    # Before tree_to_actions
    wbuffer = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    stack = [0]
    arcs = []
    deps = {5: {(5, 9): 'punct', (5, 8): 'obl', (5, 4): 'advmod', (5, 3): 'aux:pass', (5, 2): 'nsubj:pass'},
            8: {(8, 7): 'det', (8, 6): 'case'}, 0: {(0, 5): 'root'}, 2: {(2, 1): 'nmod:poss'}}

    _, gold_transitions = tree_to_actions(wbuffer, stack, arcs, deps)

    # After tree_to_actions
    assert wbuffer == [], "The result for wbuffer is not correct"
    assert stack == [0], "The result for stack is not correct"
    assert arcs == [('nmod:poss', 2, 1), ('advmod', 5, 4), ('aux:pass', 5, 3), ('nsubj:pass', 5, 2),
                    ('det', 8, 7), ('case', 8, 6), ('obl', 5, 8), ('punct', 5, 9), ('root', 0, 5)], \
        "The result for arcs is not correct"
    assert deps == {5: {(5, 9): 'punct', (5, 8): 'obl', (5, 4): 'advmod', (5, 3): 'aux:pass', (5, 2): 'nsubj:pass'},
                    8: {(8, 7): 'det', (8, 6): 'case'}, 0: {(0, 5): 'root'}, 2: {(2, 1): 'nmod:poss'}}, \
        "The result for deps is not correct"


def test_tree_to_actions_a():
    """
    Sanity check for the function tree_to_actions()
    """
    # Before tree_to_actions
    wbuffer = [5, 4, 3, 2, 1]
    stack = [0]
    arcs = []
    deps = {
        1: {
            (1, 2): 'iobj',
            (1, 5): 'obj'
        },
        5: {
            (5, 3): 'det',
            (5, 4): 'nmod'
        },
        0: {
            (0, 1): 'root'
        }
    }

    _, gold_transitions = tree_to_actions(wbuffer, stack, arcs, deps)

    # After tree_to_actions
    assert wbuffer == [], "The result for wbuffer is not correct"
    assert stack == [0], "The result for stack is not correct"
    assert gold_transitions == ['SHIFT', 'SHIFT', 'RIGHTARC_iobj', 'SHIFT', 'SHIFT', 'SHIFT', 'LEFTARC_nmod',
                                'LEFTARC_det', 'RIGHTARC_obj', 'RIGHTARC_root']


def test_tree_to_actions_c():
    """
    Sanity check for the function tree_to_actions()
    """
    # Before tree_to_actions
    wbuffer = [5, 4, 3, 2, 1]
    stack = [0]
    arcs = []
    deps = {
        0: {
            (0, 1): ''
        },
        1: {
            (1, 3): ''
        },
        3: {
            (3, 2): '',
            (3, 5): ''
        },
        5: {
            (5, 4): ''
        }
    }

    _, gold_transitions = tree_to_actions(wbuffer, stack, arcs, deps)

    # After tree_to_actions
    assert wbuffer == [], "The result for wbuffer is not correct"
    assert stack == [0], "The result for stack is not correct"
    assert gold_transitions == ['SHIFT', 'SHIFT', 'SHIFT', 'LEFTARC_', 'SHIFT', 'SHIFT', 'LEFTARC_', 'RIGHTARC_',
                                'RIGHTARC_', 'RIGHTARC_']


reverse_labels = ['SHIFT', 'RIGHTARC_punct', 'RIGHTARC_flat', 'LEFTARC_amod', 'LEFTARC_nsubj', 'LEFTARC_det',
                  'RIGHTARC_appos',
                  'RIGHTARC_obj', 'LEFTARC_case', 'RIGHTARC_nmod', 'RIGHTARC_obl', 'RIGHTARC_parataxis',
                  'RIGHTARC_root', 'LEFTARC_aux',
                  'LEFTARC_punct', 'RIGHTARC_iobj', 'LEFTARC_mark', 'RIGHTARC_acl', 'RIGHTARC_compound:prt',
                  'LEFTARC_nummod',
                  'RIGHTARC_ccomp', 'LEFTARC_aux:pass', 'LEFTARC_nsubj:pass', 'LEFTARC_compound',
                  'LEFTARC_nmod:poss', 'LEFTARC_cc',
                  'RIGHTARC_conj', 'LEFTARC_advmod', 'RIGHTARC_xcomp', 'LEFTARC_advcl', 'RIGHTARC_advmod',
                  'RIGHTARC_acl:relcl',
                  'RIGHTARC_advcl', 'LEFTARC_expl', 'RIGHTARC_nsubj', 'LEFTARC_obl', 'LEFTARC_cop',
                  'RIGHTARC_fixed', 'RIGHTARC_nummod',
                  'LEFTARC_det:predet', 'RIGHTARC_obl:npmod', 'RIGHTARC_obl:tmod', 'LEFTARC_obl:tmod',
                  'RIGHTARC_nmod:tmod',
                  'RIGHTARC_amod', 'LEFTARC_csubj', 'LEFTARC_csubj:pass', 'RIGHTARC_case', 'RIGHTARC_det',
                  'LEFTARC_obj',
                  'LEFTARC_nmod:tmod', 'LEFTARC_nmod', 'RIGHTARC_cop', 'RIGHTARC_expl', 'RIGHTARC_aux',
                  'RIGHTARC_vocative',
                  'RIGHTARC_csubj', 'LEFTARC_obl:npmod', 'RIGHTARC_nmod:npmod', 'RIGHTARC_list', 'LEFTARC_ccomp',
                  'LEFTARC_discourse',
                  'LEFTARC_parataxis', 'LEFTARC_xcomp', 'RIGHTARC_csubj:pass', 'LEFTARC_cc:preconj',
                  'RIGHTARC_flat:foreign',
                  'RIGHTARC_compound', 'LEFTARC_acl:relcl', 'RIGHTARC_discourse', 'LEFTARC_nmod:npmod',
                  'LEFTARC_acl', 'LEFTARC_vocative',
                  'LEFTARC_goeswith', 'LEFTARC_conj', 'LEFTARC_appos', 'RIGHTARC_goeswith', 'RIGHTARC_aux:pass',
                  'RIGHTARC_nsubj:pass',
                  'LEFTARC_orphan', 'LEFTARC_reparandum', 'RIGHTARC_reparandum', 'LEFTARC_list',
                  'RIGHTARC_cc:preconj', 'LEFTARC_dep',
                  'RIGHTARC_dep', 'RIGHTARC_nmod:poss', 'RIGHTARC_mark', 'RIGHTARC_cc']


def test_action_to_tree_shift():
    """
    Sanity check for the function action_to_tree()
    """
    # Before action
    tree = {}
    predictions = np.array(
        [[8.904456, 2.1306312, -0.6716528, -0.37662476, -0.01239625, -3.3660867, -2.1345713, 1.4581618,
          -0.1688145, -0.61321, 0.40860286, -2.7569351, -0.69548404, -0.7809651, 0.7595304, -2.770731,
          -0.97373027, -2.70085, -0.26645675, -1.2353135, -1.4289687, -1.3272284, -2.4956157, -1.0178847,
          -1.7484616, 1.7610879, 0.301237, -0.71727145, -1.9370077, -1.3722429, 0.9516849, -2.6749346,
          -1.4604743, -1.6903474, -2.5261753, -0.88417345, -0.50328434, -0.21296862, -3.4296887, -3.3282495,
          -4.300956, -2.12365, -3.3637137, -5.570282, -3.8983932, -3.0985348, -5.818429, -1.5155774,
          -3.4247532, -2.7098398, -4.799152, -4.020282, -3.5505116, -2.7114115, -4.1488724, -4.7484784,
          -4.0955606, -2.994336, -4.9744525, -4.3390574, -2.782462, -4.615161, -4.6250424, -4.4105268,
          -4.856515, -3.5684056, -4.6808653, -4.882898, -4.3673973, -5.379696]])

    wbuffer = [4, 3, 2, 1]
    stack = [0]
    arcs = []

    # Perform action
    action_to_tree(tree, predictions, wbuffer, stack, arcs, reverse_labels)

    # After action (the action is SHIFT for this step)
    assert not tree, "The tree should be {} after the SHIFT"
    assert wbuffer == [4, 3, 2], "wbuffer should be [4,3,2] after the SHIFT"
    assert stack == [0, 1], "stack should be [0, 1] after the SHIFT"
    assert arcs == [], "arcs should be [] after the SHIFT"


def test_action_to_tree_rightarc():
    """
    Sanity check for the function action_to_tree()
    """
    # Before action
    tree = {}
    predictions = np.array(
        [0, 0, 5, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0])

    wbuffer = [4, 3]
    stack = [0, 2, 1]
    arcs = []

    # Perform action
    action_to_tree(tree, predictions, wbuffer, stack, arcs, reverse_labels)

    # After action (the action is SHIFT for this step)
    assert tree == {1: (2, 'flat')}
    assert wbuffer == [4, 3]
    assert stack == [0, 2]
    assert arcs == [('flat', 2, 1)]


def test_action_to_tree_leftarc():
    """
    Sanity check for the function action_to_tree()
    """
    # Before action
    tree = {}
    predictions = np.array(
        [0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0])

    wbuffer = [4, 3]
    stack = [0, 2, 1]
    arcs = []

    # Perform action
    action_to_tree(tree, predictions, wbuffer, stack, arcs, reverse_labels)

    # After action (the action is SHIFT for this step)
    assert tree == {2: (1, 'amod')}
    assert wbuffer == [4, 3]
    assert stack == [0, 1]
    assert arcs == [('amod', 1, 2)]


def test_action_to_tree_invalid():
    tree = {}
    predictions = np.array(
        [0, 0, 5, 100, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0])

    wbuffer = [4, 3]
    stack = [0,1]
    arcs = [('amod', 1, 2)]

    # Perform action
    action_to_tree(tree, predictions, wbuffer, stack, arcs, reverse_labels)

    # After action (the action is SHIFT for this step)
    assert tree == {1: {0, 'flat'}}
    assert wbuffer == [4]
    assert stack == [0, 1, 3]
    assert arcs == [('amod', 1, 2)]
