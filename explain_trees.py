import sklearn


def explain_tree_ensemble_prediction(model, example):
    pass


def explain_tree_prediction(model, example):
    """
    Args:
        model: A decision tree model.
        example (pandas.Series): A point to generate a prediction for.

    Returns:
        A list of branch selection rules used for generating the example's prediction, ordered
        from the root to the leaf.
        (feature_id, split_value, rule_satisfied, feature_continuous)
    """
    tree = model.tree_
    criteria = []
    node_id = 0
    while tree.children_left[node_id] != sklearn.tree._tree.TREE_LEAF:
        feature_id = tree.feature[node_id]
        split_value = tree.threshold[node_id]

        example_value = example.iloc[feature_id]
        continuous = True
        rule_satisfied = is_rule_satisfied(split_value, example_value, continuous)

        if rule_satisfied:
            child_node_id = tree.children_left[node_id]
        else:
            child_node_id = tree.children_right[node_id]

        # impurity is scaled differently for gini vs. entropy
        decrease_impurity = tree.impurity[node_id] - tree.impurity[child_node_id]

        # include comparison of concentrations of different values
        criteria.append((feature_id, split_value, example_value, rule_satisfied, decrease_impurity))
        node_id = child_node_id

    assert tree.children_right[node_id] == sklearn.tree._tree.TREE_LEAF

    return criteria


def prediction_descriptions(model, example):
    explanation = explain_tree_prediction(model, example)
    return [tree_prediction_str(example, decision) for decision in explanation]


def tree_prediction_str(example, decision):
    feature_name = example.index[decision[0]]
    if decision[3]:
        operator = '<='
    else:
        operator = '>'

    return '%s=%f %s %f decreases impurity %f' % \
        (feature_name, decision[2], operator, decision[1], decision[4])


def is_rule_satisfied(split_value, example_value, continuous):
    # if continuous:
    #     return example_value > split_value
    # else:
    #     return example_value != split_value
    return split_value > example_value
