import abc
from math import exp, log
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score
from gbdt.tree import Tree


class ClassificationLossFunction(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def initialize(self, f, data):
        """初始化f_{0}"""

    @abc.abstractmethod
    def calculate_residual(self, data, iter):
        """计算残差"""

    @abc.abstractmethod
    def update_f_value(self, data, tree, iter, learning_rate):
        """更新f_{m}"""

    @abc.abstractmethod
    def update_leaf_values(self, targets, y):
        """更新叶子节点的预测值"""

    @abc.abstractmethod
    def get_train_loss(self, y, f):
        """计算训练损失"""


class BinomialDeviance(ClassificationLossFunction):

    def initialize(self, data):
        pos = sum(data['label'] == 1)
        neg = len(data) - pos
        f_0 = log(pos/neg)
        data['f_0'] = f_0
        return f_0

    def calculate_residual(self, data, iter):
        r_name = 'r_' + str(iter)
        f_prev_name = 'f_' + str(iter - 1)
        data[r_name] = data['label'] - 1 / (1 + data[f_prev_name].apply(lambda x: exp(-x)))
        
    def update_f_value(self, data, tree, iter, learning_rate):
        f_prev_name = 'f_' + str(iter - 1)
        f_m_name = 'f_' + str(iter)
        for leaf_node in tree.leaf_nodes:
            data.loc[leaf_node.data_index, f_m_name] = data.loc[leaf_node.data_index, f_prev_name] + \
                                                       learning_rate * leaf_node.predict_value
                            
    def update_leaf_values(self, targets, y):
        numerator = targets.sum()
        if numerator == 0:
            return 0.0
        denominator = ((y - targets) * (1 - y + targets)).sum()
        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return numerator / denominator

    def get_train_loss(self, y, f):
        loss = -2.0 * ((y * f) - f.apply(lambda x: exp(1+x))).mean()
        return loss

# 基于本地数据跑n颗树
def get_predict_value(data, trees_in_matrix):
    n_trees = len(trees_in_matrix)
    for t in range(n_trees):
        tree_in_vector = trees_in_matrix[t]
        # 将vector树转化为dic
        tree_nodes = {}
        n_nodes = int(tree_in_vector[0])
        tree_in_vector = tree_in_vector[1:].tolist()
        for node in range(n_nodes):
            tree_nodes[node] = {'is_leaf': tree_in_vector[node*7+1],
                                'split_feature': tree_in_vector[node*7+2],
                                'split_value': tree_in_vector[node*7+3],
                                'left_node_id': tree_in_vector[node*7+4],  
                                'right_node_id': tree_in_vector[node*7+5],
                                'predict_value': tree_in_vector[node*7+6]}
        # 跑本地数据
        t_name = 't_' + str(t+1)
        data[t_name] = None
        for i in data.index:
            next_node = tree_nodes[len(tree_nodes)-1] # root_node
            while next_node['is_leaf'] == 0:
                if data.loc[i,data.columns[next_node['split_feature']]] < next_node['split_value']:
                    next_node = tree_nodes[next_node['left_node_id']]
                else:
                    next_node = tree_nodes[next_node['right_node_id']]
            data.loc[i,t_name] = next_node['predict_value']
        

class GBDT:

    def __init__(self, learning_rate, max_iter, max_depth, min_samples_split=2, loss_type='binary-classification'):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss_type = loss_type
        self.loss = None

        # self.trees = {}
        self.trees_in_matrix = []
        
    def fit(self, datasets):
        if self.loss_type == 'binary-classification':
            self.loss = BinomialDeviance()

        m = 1
        for i in range(self.max_iter):
            for j in range(len(datasets)):
                data = datasets[j].copy()
                # 删除label，得到特征名称
                features = list(data.drop('label', axis = 1).columns)

                # 初始化f_{0}
                self.f_0 = self.loss.initialize(data)
                
                # 计算f_{m-1}
                if m == 1:
                    pass
                else:
                    # 本地数据跑所有树
                    get_predict_value(data, self.trees_in_matrix)

                    # f_{m-1} = f_{0} + T_{1} + ... + T_{m-1}
                    f_m_name = 'f_' + str(m-1)
                    data[f_m_name] = data['f_0']
                    for t in range(1,m):
                        t_m_name = 't_' + str(t)
                        data[f_m_name] += self.learning_rate * data[t_m_name]

                # 计算残差r_{m}
                self.loss.calculate_residual(data, m)

                # 拟合残差学习一个回归树
                target_name = 'r_' + str(m)
                tree = Tree(data, self.max_depth, self.min_samples_split, features, self.loss, target_name)

                # 传递树的class
                # self.trees[m] = tree
                
                # 传递树的matrix - PyTorch
                n_nodes = int(len(tree.tree_in_vector)/7)
                tree_with_count_prefix = [n_nodes] + tree.tree_in_vector
                print(tree_with_count_prefix)
                n_trees = len(self.trees_in_matrix)
                if n_trees == 0: 
                    self.trees_in_matrix = torch.tensor([tree_with_count_prefix])
                else:
                    pre_col = self.trees_in_matrix.shape[1]
                    new_col = n_nodes*7+1 
                    max_col = new_col if new_col > pre_col else pre_col
                    temp = torch.zeros(n_trees+1, max_col)
                    temp[0:n_trees,0:pre_col] = self.trees_in_matrix
                    temp[-1,0:new_col] = torch.tensor(tree_with_count_prefix)
                    self.trees_in_matrix = temp

                # f_{m} = f_{m-1} + T_{m} 
                self.loss.update_f_value(data, tree, m, self.learning_rate)
                
                # 计算训练损失
                train_loss = self.loss.get_train_loss(data['label'], data['f_' + str(m)])
                print('iter%d party%d tree%d: train loss=%f \n' % (i+1, j+1, m, train_loss))

                m += 1
        
        print(self.trees_in_matrix)

    def predict(self, data, type):
        get_predict_value(data, self.trees_in_matrix)
        data['predict_proba'] = data.iloc[:,-1].apply(lambda x: 1 / (1 + exp(-x)))
        data['predict_label'] = data['predict_proba'].apply(lambda x: 1 if x >= 0.5 else 0)

        if type == 'proba':
            return(data['predict_proba'])
        elif type == 'label':
            return(data['predict_label'])

def evaluation(y_test, y_pred, y_proba):
    [[TN, FP], [FN, TP]] = confusion_matrix(y_test, y_pred)
    print([[TN, FP], [FN, TP]])
    accuracy = (TP + TN) * 1.0 / (TP + FP + TN + FN)
    precision = TP * 1.0 / (TP + FP) if (TP + FP) != 0 else np.nan
    recall = TP * 1.0 / (TP + FN) if (TP + FN) != 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall)
    auc = roc_auc_score(y_test, y_proba)
    sen = TP * 1.0 / (TP + FN) if (TP + FN) != 0 else np.nan
    spe = TN * 1.0 / (TN + FP) if (TN + FP) != 0 else np.nan
    res = pd.DataFrame(columns=['Accuracy', 'Precision(ppv)', 'Recall', 'F1-score', 'ROC-AUC', 'Sensitivity', 'Specificity'])
    res.loc[0] = [accuracy, precision, recall, f1, auc, sen, spe]
    print(res)
    return(res)
        