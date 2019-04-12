import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score
class KNNClassifier:
    def __init__(self,k):
        # 初始化kNN分类器
        assert k >= 1,"k must be valid"
        self.k=k
        self._X_train=None
        self._y_train=None
    def fit(self,X_train,y_train):
        # 根据训练数据集X_train和y_train训练KNN分类器
        self._X_train=X_train
        self._y_train=y_train
        return self
    def predict(self,X_predict):
        y_predict=[self._predict(x) for x in X_predict]
        return np.array(y_predict)
    def _predict(self,x):
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        vote = Counter(topK_y)
        return vote.most_common(1)[0][0]
    def score(self,X_test,y_test):
        y_predict=self.predict(X_test)
        return  accuracy_score(y_test,y_predict)

    def __repr__(self):
        return "KNN(k=%d)" %self.k
