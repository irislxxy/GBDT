import pandas as pd
from gbdt.model import GBDT
from gbdt.model import evaluation

if __name__ == '__main__':
    data1 = pd.read_csv('./data/train1.csv')
    data2 = pd.read_csv('./data/train2.csv')
    datasets = [data1,data2]
    gbdt = GBDT(learning_rate=0.5, max_iter=5, max_depth=3, min_samples_split=2, loss_type='binary-classification')
    gbdt.fit(datasets)
    
    test_data = pd.read_csv('./data/test.csv')
    y_pred = gbdt.predict(test_data, 'label')
    y_proba = gbdt.predict(test_data, 'proba') 
    evaluation(test_data['label'], y_pred, y_proba)