import numpy as np
import pandas as pd

import myDesicionTree import DecisionTree, gini, Node

class randomforest(object):
    def __init__(self, n_trees, criterion, max_high):
        self.trees = None
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_high = max_high
    
    def fit(self, X_train, y_train, is_continuous):
        '''
        拟合数据
        输入X，y
        输出森林
        调用函数：决策树，拟合决策树
        '''
        self.trees = []
        m = X_train.shape[0]
        for _ in range(self.n_trees):
            indices = np.random.choice(m,m,replace=True)
            X_train_i = X_train.iloc[indices,:]
            y_train_i = y_train.iloc[indices]
            tree = DecisionTree(self.criterion,self.max_high)
            tree.fit(X_train_i, y_train_i, is_continuous)
            self.trees.append(tree)
        
    def predict(self, X):
        '''
        预测
        输入X
        输出预测值
        '''
        trees_predictions = [[t.predict_by_one(x) for t in self.trees] for x in X.values]
        return self.vote(trees_predictions)

    def vote(self, predictions):
        '''
        多数表决
        输入predictions
        输出最多的预测结果
        '''
        return [np.argmax(np.bincount(x)) for x in predictions]
