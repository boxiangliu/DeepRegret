import pandas as pd
import numpy as np
import random

#### LOAD DATA ###################################
def one_hot(seq):
    encoded_seq = np.zeros((4, len(seq)))
    for pos in range(len(seq)):
        nt = seq[pos]
        if nt == "A":
            encoded_seq[0, pos] = 1
        if nt == "C":
            encoded_seq[1, pos] = 1
        if nt == "G":
            encoded_seq[2, pos] = 1
        if nt == "T":
            encoded_seq[3, pos] = 1
    return encoded_seq

# ratios is a 3-tuple or 3-list that gives training:val:test
def partition(data, ratios):
    ratio_tvt = np.array(ratios)
    pct_tvt = ratio_tvt/float(sum(ratio_tvt))

    partition_vec = np.zeros(len(data))
    partition_vec[int(len(data)*pct_tvt[:1]):] = 1
    partition_vec[int(len(data)*pct_tvt[:2].sum()):] = 2
    
    random.seed(42)
    random.shuffle(partition_vec)
    
    return [data[partition_vec==i] for i in range(3)]

# for 1 epoch: (will throw away any data that does not fit in evenly)
# yields none on when there is no more data
def batch_generator(data, batch_size):
    data_len = len(data)

    num_batches = int(data_len/batch_size)
    partition_range = np.arange(data_len)
    random.shuffle(partition_range)
    for i in range(num_batches):

        batch_list = partition_range[i*batch_size:(i+1)*batch_size]
        
        X1 = data['one_hot_sequence'].iloc[batch_list]
        X2 = data['reg_exp'].iloc[batch_list]
        Y = data['expression'].iloc[batch_list]

        yield np.transpose(np.stack(X1.as_matrix(),0),(0,2,1)), np.stack(X2.as_matrix(),0), np.hstack(Y.as_matrix())



#########
reg_names = pd.read_table("../data/reg_names_R.txt", names = ["UID"])

expr_data = pd.read_table("../data/complete_dataset.txt").fillna(0).drop("NAME", axis=1).drop("GWEIGHT", axis=1)

target_expr_data = pd.melt(expr_data, id_vars="UID", var_name="experiment", value_name="expression")

promoters = pd.read_table("../data/yeast_promoters.txt", names=["UID", "sequence"])
promoters.loc[:, "one_hot_sequence"] = [one_hot(seq) for seq in promoters.loc[:, "sequence"]]

reg_data = pd.merge(reg_names, expr_data, on="UID", how="inner").drop("UID", axis=1)

reg = pd.DataFrame()
for col in range(len(reg_data.columns)):
    data = np.array([exp_level for exp_level in reg_data.iloc[:, col]])
    reg = reg.append(pd.DataFrame({"experiment": reg_data.columns[col], "reg_exp": [data]}))

data_complete = pd.merge(promoters, target_expr_data, on="UID", how="inner").merge(reg, on="experiment", how="inner")

train, val, test = partition(data_complete, (60,10,30))

def getTrain():
    return train

def getVal():
    return val

def getTest():
    return test


############
# def generate_train_batch(batch_size):
#     return batch_generator(data_train, batch_size)

# def generate_val_batch(batch_size):
#     return batch_generator(data_val, batch_size)

# def generate_test_batch(batch_size):
#     return batch_generator(data_test, batch_size)
