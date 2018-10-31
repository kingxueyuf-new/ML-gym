import pandas as pd
from collections import OrderedDict

c_index = 0
label_to_class_idx = OrderedDict()
query_to_class_idx = OrderedDict()


def read_csv(abs_path) -> OrderedDict():
    df = pd.read_csv(abs_path)
    print(df)
    for i in range(len(df)):
        query = df['query'][i:i + 1].values[0]
        label = df['label'][i:i + 1].values[0]
        if label not in label_to_class_idx.keys():
            global c_index
            label_to_class_idx[label] = c_index
            c_index += 1
        query_to_class_idx[query] = label_to_class_idx[label]

    print(label_to_class_idx)
    print(query_to_class_idx)
    return query_to_class_idx

if __name__ == '__main__':
    read_csv('../../LSTM classifier label - Sheet1.csv')
