import abc
from math import exp, log
import pandas as pd
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
    def update_f_value(self, data, trees, iter, learning_rate):
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

    def update_f_value(self, data, trees, iter, learning_rate):
        f_prev_name = 'f_' + str(iter - 1)
        f_m_name = 'f_' + str(iter)
        data[f_m_name] = data[f_prev_name]
        for leaf_node in trees[iter].leaf_nodes:
            data.loc[leaf_node.data_index, f_m_name] += learning_rate * leaf_node.predict_value

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

class AbstractBaseGradientBoosting(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def fit(self, datasets):
        pass

    def predict(self, data):
        pass


class BaseGradientBoosting(AbstractBaseGradientBoosting):

    def __init__(self, loss, learning_rate, max_iter, max_depth, min_samples_split=2):
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = None
        self.trees = {}
        self.f_0 = {}

    def fit(self, datasets):
        m = 1
        for i in range(self.max_iter):
            for j in range(len(datasets)):
                data = datasets[j].copy()
                print(data.columns)

                # 删除label，得到特征名称
                self.features = list(data.drop('label', axis = 1).columns)
                print(self.features)

                # 初始化f_{0}
                self.f_0 = self.loss.initialize(data)

                # 计算f_{m-1}
                if m == 1:
                    pass
                else:
                    # f_{m-1} = T_{1} + ... + T_{m-1}
                    f_m_name = 'f_' + str(m-1)
                    data[f_m_name] = data['f_0']
                    for iter in range(1,m):
                        for leaf_node in self.trees[iter].leaf_nodes:
                            data.loc[leaf_node.data_index, f_m_name] += self.learning_rate * leaf_node.predict_value

                # 计算残差r_{m}
                self.loss.calculate_residual(data, m)
                print(data.columns)
                # 拟合残差学习一个回归树
                target_name = 'r_' + str(m)
                tree = Tree(data, self.max_depth, self.min_samples_split, self.features, self.loss, target_name)
                self.trees[m] = tree
                # f_{m} = f_{m-1} + T_{m} 
                self.loss.update_f_value(data, self.trees, m, self.learning_rate)
                # 计算训练损失
                train_loss = self.loss.get_train_loss(data['label'], data['f_' + str(m)])
                print('iter%d party%d tree%d: train loss=%f \n' % (i+1, j+1, m, train_loss))

                m += 1
            

class GradientBoostingBinaryClassifier(BaseGradientBoosting):
    def __init__(self, learning_rate, max_iter, max_depth, min_samples_split=2):
        super().__init__(BinomialDeviance(), learning_rate, max_iter, max_depth, min_samples_split)

    def predict(self, data, type):
        data['f_0'] = self.f_0
        for iter in range(1, len(self.trees)+1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        data['predict_proba'] = data[f_m_name].apply(lambda x: 1 / (1 + exp(-x)))
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
        