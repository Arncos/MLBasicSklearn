import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
iris = load_iris()
test_idx = [0,50,100]
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)
# print train_target
# print train_data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
# print test_target
# print test_data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
# print clf.predict(test_data)
dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    impurity=False)
graph = graphviz.Source(dot_data)
graph.render("iris")
print test_data[0],test_target[0]
print iris.feature_names, iris.target_names