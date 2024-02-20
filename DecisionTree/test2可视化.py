import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

#从CSV文件加载原始数据
raw_data = pd.read_csv('f:/breast-cancer-wisconsin.data', header=None)
#从数据中提取特征和标签
data = raw_data.values
features = data[:, 1:-1]
labels = data.iloc[:, -1].values

#将数据拆分为训练集和测试集
train_attributes, test_attributes, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)

#建立决策树模型
clf = DecisionTreeClassifier(criterion="entropy", random_state=0, splitter="random")
clf.fit(train_attributes, train_labels)
accuracy = clf.score(test_attributes, test_labels)
print("Accuracy:", accuracy)

#显示
tree_rules = export_text(clf, feature_names=list(data.columns))
print(tree_rules)

#可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=list(data.columns), class_names=["良性", "恶性"])
plt.show()
