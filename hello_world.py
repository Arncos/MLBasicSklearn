from sklearn import tree
import graphviz
features = [[140, 1], [130, 1], [150, 0], [170, 0], [180, 0]]
labels = [0,0,1,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print clf.predict([[130,1]])