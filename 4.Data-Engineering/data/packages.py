# import libraries
import pandas as pd
import numpy as np

# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def multioutput_classification_report(y_test, y_pred):
    
    # accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
    # categories_name = list(y_test.columns)
    
    # for i in range(0, len(y_test.columns)):
 
    #     accuracy_list.append(accuracy_score(y_test.values[:, i], y_pred[:, i]))
    #     precision_list.append(precision_score(y_test.values[:, i], y_pred[:, i], average='weighted'))
    #     recall_list.append(recall_score(y_test.values[:, i], y_pred[:, i], average='weighted'))
    #     f1_list.append(f1_score(y_test.values[:, i], y_pred[:, i], average='weighted'))
    
    # df = pd.DataFrame([accuracy_list,precision_list,recall_list,f1_list],
    #                             columns=categories_name,
    #                             index=['accuracy_score',
    #                                    'precision_score',
    #                                    'recall_score',
    #                                    'f1_score'])

    f1_list = []
    categories_name = list(y_test.columns)
    
    for i in range(0, len(y_test.columns)):
 
        f1_list.append(f1_score(y_test.values[:, i], y_pred[:, i], average='weighted'))
    
    df = pd.DataFrame([f1_list],columns=categories_name,index=['f1_score'])

    df = df.T
    
    return df