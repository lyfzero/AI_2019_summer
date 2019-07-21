import numpy as np
import pandas as pd

class Node(object):
    def __init__(self):
        self.leaf_num = 0
        self.is_leaf = False
        self.leaf_pred = None
        self.split_feature = None
        self.feature_index = None
        self.is_continuous = None
        self.split_feature_val = None
        self.child = {}
        self.high = -1


def gini(y):
    '''
    计算基尼值
    输入：y
    返回基尼值
    '''
    p = y.value_counts() / y.shape[0]
    gini =  1 - np.sum(p ** 2)
    return gini

class DecisionTree(object):
    def __init__(self, criterion, max_high):
        self.criterion = criterion
        self.pruning = None
        self.tree = None
        self.features = None
        self.is_continuous = None
        self.max_high = max_high

    def fit(self, X_train, y_train, is_continuous):
        '''
        拟合数据
        输入：X, y
        输出树
        调用函数：pre_pruning(输入训练数据) \post_pruning（输入验证数据） \create_tree（输入训练数据）

        '''
        self.features = list(X_train.columns)
        self.is_continuous = is_continuous
        self.tree = self.create_tree(X_train, y_train)


    def create_tree(self, X, y):
        '''
        生成树的关键函数（递归）
        输入：X，y
        返回树
        调用函数：类node（生成节点，根、内点、叶节点） 、判断y的取值是否唯一 、判断X是否为空 、 y最多次数的取值 、 
                选择最合适的特征切分choose_best_feature_to_split(输入X，y返回最合适的特征名和不纯度(连续特征的不纯度还需有最合适的切分位置))
                若为特征的取值为离散值，对每个取值生成子树？（不会太多吗），调用create_tree(输入去掉特征的X，和y，满足特征值相等)
                若特征的取值为连续值，按照切分位置将数据分为左子树和右子树
        '''
        tree = Node()
        tree.leaf_num = 0
        if y.nunique() == 1:
            tree.is_leaf = True
            tree.leaf_pred = y.values[0]
            tree.leaf_num += 1
            tree.high = 0
            return tree
        
        if X.empty or tree.high > self.max_high or X.duplicated(keep=False).sum()==X.shape[0]:
            tree.is_leaf = True
            tree.leaf_pred = y.value_counts().index[0]
            tree.leaf_num += 1
            tree.high = 0
            return tree
        print(X.shape,X.columns)

        split_feature, impurity = self.choose_best_feature_to_split_gini(X, y)

        tree.split_feature = split_feature
        tree.feature_index = self.features.index(split_feature)
        tree.impurity = impurity[0]

        feature_values = X.loc[:, split_feature]

        if self.is_continuous[tree.feature_index] == 1:
            tree.is_continuous = True
            tree.split_feature_val = impurity[1]
            sub_X = X.copy().drop(split_feature, axis=1)
            low_tree = '>= {:.3f}'.format(tree.split_feature_val)
            high_tree = '< {:.3f}'.format(tree.split_feature_val)
            low_rows = feature_values >= impurity[1]
            high_rows = feature_values <impurity[1]
            tree.child[high_tree] = self.create_tree(sub_X[high_rows],y[high_rows])
            tree.child[low_tree] = self.create_tree(sub_X[low_rows], y[low_rows])
            tree.leaf_num += tree.child[high_tree].leaf_num +tree.child[low_tree].leaf_num
            tree.high = max(tree.child[high_tree].high, tree.child[low_tree].high)

        elif self.is_continuous[tree.feature_index] == 0:
            tree.is_continuous = False
            high = -1
            feature_values_unique = feature_values.unique()
            sub_X = X.copy().drop(split_feature, axis=1)
            for value in feature_values_unique:
                tree.child[value] = self.create_tree(sub_X[feature_values == value], y[feature_values == value])
                tree.leaf_num += tree.child[value].leaf_num
                high = max(high, tree.child[value].high) 
            tree.high = high + 1
        
        return tree
        

    def choose_best_feature_to_split_gini(self, X, y):
        '''
        根据Gini选出最合适的切分特征
        输入：X， y
        返回最合适的切分特证名和不纯度gini指标（若为连续值还加上最好的切分值）
        调用函数：计算gini值gini_index（输入X在某特征下的数据，y，是否为连续值，返回gini指数）
        '''
        best_gini = [float('inf')]
        split_feature = None
        features = X.columns
        for feature in features:
            is_continuous = self.is_continuous[self.features.index(feature)]
            gini_index = self.get_gini_index(X[feature], y, is_continuous)
            print(gini_index)
            if gini_index[0] < best_gini[0]:
                split_feature = feature
                best_gini = gini_index
        
        return split_feature, best_gini


    
    @staticmethod
    def get_gini_index(feature, y, is_continuous):
        '''
        计算基尼指数，若为连续值，选择基尼指数最小的点作为分割点
        输入：各特征值，y
        返回最小的gini（和分割点）
        调用函数：计算gini值gini（输入y，返回gini值）
        '''
        m = feature.shape[0]
        feature_values_unique = sorted(feature.unique())
        if is_continuous:
            split_points = [(feature_values_unique[i]+feature_values_unique[i+1])/2 for i in range(len(feature_values_unique)-1)] 
            best_gini = float('inf')
            best_split_point = None
            for split_point in split_points:
                part_low = y[feature <= split_point]
                part_high = y[feature > split_point]
                gini_index = len(part_low) / m * gini(part_low) + len(part_high) / m * gini(part_high)

                if gini_index < best_gini:
                    best_gini = gini_index
                    best_split_point = split_point
            return [best_gini, best_split_point]

        else:
            best_gini = 0
            for val in feature_values_unique:
                part = y[feature == val]
                best_gini += len(part) / m * gini(part)
            return [best_gini]
        
    
    def predict(self, X):
        '''
        预测
        输入：X
        返回X对应的预测结果
        调用函数：一条数据一条数据预测predic_for_one(输入一条数据x，返回其预测结果)
        用循环写，递归容易栈溢出
        '''
        return X.apply(self.predict_by_one, axis=1)

    def predict_by_one(self, x):
        '''
        输入x
        输出预测结果
        '''
        tree = self.tree
        while not tree.is_leaf:
            if tree.is_continuous:
                if x[tree.feature_index] >= tree.split_feature_val:
                    tree = tree.child['>= {:.3f}'.format(tree.split_feature_val)]
                else:
                    tree = tree.child['< {:.3f}'.format(tree.split_feature_val)]
            else:
                tree = tree.child[x[tree.feature_index]]
        return tree.leaf_pred
    

