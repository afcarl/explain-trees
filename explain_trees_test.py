import unittest

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

import explain_trees


class ExplainTreesTestCase(unittest.TestCase):
    def test_explain_tree_prediction(self):
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        # model = GradientBoostingClassifier(n_estimators=10, max_depth=3).fit(df, labels)
        model = DecisionTreeClassifier().fit(df, iris.target)

        explanation = explain_trees.explain_tree_prediction(model, df.iloc[0])
        print(explanation)
