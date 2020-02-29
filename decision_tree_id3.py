import pickle as pkl
import argparse
import csv
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import datetime

# from autograder_basic.py
class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self,filename):
        obj = open(filename,'wb')
        pkl.dump(self,obj)

nodes = 0

# Decision tree id3 algorithm
def decision_tree_id3(T_data, rem_features):
    global nodes
    nodes= nodes+1

    # base case where all T_data are positive
    #print("id3")

    uniquelabels = T_data[last].unique()
    label=""
    # if there is only one value present for the feature
    if(len(uniquelabels)<=1 ):
        if(1 in T_data[last]):
            label="T"
        else:
            label="F"
        root = TreeNode(data=label, children=[])
        return root
    if (len(rem_features)==0 ):
        return get_root(T_data)

    #get the best feature based on which tree can be split (The node with highest info gain)
    best_feature = get_root_feature(T_data,  rem_features)

    pvalue = chi2_splitting(T_data, best_feature)

    if pvalue > float(args['p']):
        return get_root(T_data)


    rem_features.remove(best_feature)
    root = TreeNode(data=str(best_feature + 1))

    # calling id3 recursively for each child of current node
    for elem in range(5):
        new_T_data = T_data[T_data[best_feature] == elem + 1]
        new_T_data = new_T_data.drop(new_T_data.columns[list(new_T_data.columns).index(best_feature)], axis=1)
        root.nodes[elem] = decision_tree_id3(new_T_data, rem_features)

    return root

def get_root(T_data):
    value_counts = T_data[last].value_counts()
    # more positive T_data than negative T_data
    if value_counts[1] > value_counts[0]:
        label='T'
    else:
        label='F'

    return TreeNode(data=label,children=[])

# this function returns the feature with maximum gain (least entropy)
def get_root_feature(T_data, attr):
    attr_entropies = {}


    # for every feature, calculate entropy
    for feature in attr:
        attr_entropies[feature] = compute_entropy(T_data,feature)

    # returning feature with minimum entropy
    best = min(attr_entropies.items(), key=lambda x: x[1])

    return best[0]

def compute_entropy(T_data,feature):
    values = T_data[feature].unique()

    probs = []
    entr = []

    tc = len(T_data)

    for elem in values:
        temp = T_data.loc[T_data[feature] == elem]
        total_count = len(temp)

        attr_value_cnts = temp[last].value_counts()

        if len(temp[last].unique()) >1 :
            x = attr_value_cnts[1] / float(total_count)
            y = attr_value_cnts[0] / float(total_count)

            entropy = ((x * np.log2(x)) + (y * np.log2(y))) * (-1)

            probs.append(total_count/tc)
            entr.append(entropy)
        else:
            probs.append(total_count/tc)
            entr.append(0)

    entropy = 0
    # summing entropies for all  values in the current feature
    for i in range(len(entr)):
        entropy = entropy + (probs[i] * entr[i])

    entropy = entropy
    return entropy

#  this function calculates p-value for a particular feature
#  and decides whether to stop or not
def chi2_splitting(T_data, best_node):
    N= float(len(T_data))

    counts=T_data[last].value_counts()

    positives = getValue(counts,1)
    negetives = getValue(counts,0)

    Svalue = 0.0
    pi = 0.0
    ni = 0.0
    pi_1=0.0
    ni_1=0.0

    for elem in T_data[best_node].unique():
        temp = T_data[T_data[best_node] == elem]
        value_counts = temp[last].value_counts()
        T_i = len(temp)
        pi = getValue(value_counts,1)
        ni = getValue(value_counts,0)
        pi_1 = positives * T_i / N
        ni_1 = negetives * T_i / N

        Svalue = Svalue + np.square(pi_1 - pi) / float(pi_1) + np.square(ni_1 - ni) / float(ni_1)

    p_value = 1 - stats.chi2.cdf(Svalue,len(T_data[best_node].unique()) - 1)

    return p_value


# from autograder_basic.py
def evaluate_datapoint(root,datapoint):
    if root.data == 'T': return 1
    if root.data =='F': return 0
    return evaluate_datapoint(root.nodes[datapoint[int(root.data)-1]-1], datapoint)

def predict_labels(root,Test_data):
    print("Predicting Labels...")
    predictions = []
    for i in range(0, len(Test_data)):
        prediction = evaluate_datapoint(root, Test_data[i])
        predictions.append([prediction])
    return predictions

def save_predictions(predictions,file):
    with open(file, "wb") as f:
        out_file = csv.writer(f)
        out_file.writerows(predictions)
    print("predictions saved",len(predictions))

def getValue(value_counts,i):
    if i in value_counts.index.values:
        p = float(value_counts[i])
        return p
    else:
        return 0.0

sys.setrecursionlimit(5000)
arg_parse = argparse.ArgumentParser()
arg_parse.add_argument('-p', required=True)
arg_parse.add_argument('-f1', required=True)
arg_parse.add_argument('-f2', required=True)
arg_parse.add_argument('-o', required=True)
arg_parse.add_argument('-t', required=True)

args = vars(arg_parse.parse_args())
args = vars(arg_parse.parse_args())

pvalue=args['p']
train_dataset_file=args['f1']
train_data_label_file = args['f1'].split('.')[0] + '_label.csv'
test_dataset_file=args['f2']
output_file=args['o']
decision_tree_op=args['t']

Training_data=pd.read_csv(train_dataset_file, delim_whitespace=True,header=None)
Test_data=pd.read_csv(test_dataset_file, delim_whitespace=True,header=None)
Test_data=np.array(Test_data)
labels=pd.read_csv(train_data_label_file,delim_whitespace=True,header=None)
training_set = pd.concat([Training_data, labels], axis=1, ignore_index=True)
last = len(training_set.columns) - 1

rem_features= list()
for elem in training_set.columns:
	rem_features.append(elem)
rem_features.remove(last)
print("Trining the model for decision tree id3 algorithm...")
root = decision_tree_id3(training_set, rem_features)

obj = open(decision_tree_op, 'wb')
pkl.dump(root, obj)

labels=predict_labels(root,Test_data)
save_predictions(labels,output_file)
print("number of nodes",nodes)