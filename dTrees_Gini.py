import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Node:
    def __init__(self, data):
      self.left = None
      self.middle = None
      self.right = None
      self.data = data

dataset = pd.read_csv('US_dataset.csv')

train_accuracy = []
test_accuracy = []

def findMajority(D):
    values, counts = np.unique(D[:, -1], return_counts=True)
    majority = values[np.argmax(counts)]
    return majority

def decisionTree(D, L): 
    N = Node(None)

    #Stopping Criteria
    if len(np.unique(D[:, -1])) == 1:
        N.data = D[0, -1]
        return N 
    if len(L) == 0:
        N.data = findMajority(D)
        return N
    
    #Extra Credit 2
    # values, counts = np.unique(D[:, -1], return_counts=True)
    # if np.max(counts) >= 0.85 * len(D[:, -1]):
    #     N.data = values[np.argmax(counts)]
    #     return N
    
    attr = giniCriteria(D, L)
    N.data = attr
    attrIndex = np.where(L == attr)[0][0]
    L = np.delete(L, attrIndex)

    V = [0, 1, 2]
    for v in V: 
        D_v = D[D[:, attrIndex] == v]
        if len(D_v) == 0: 
            T= Node(findMajority(D))
        else: 
            D_v = np.delete(D_v, attrIndex, axis=1)
            T = decisionTree(D_v, L)
        if v == 0:
            N.left = T
        elif v == 1:
            N.middle = T
        else:
            N.right = T
    return N

def giniCriteria(D, L): 
    gini_values = []
    for i in range(len(L)): 
        gini_avg = 0
        V = np.unique(D[:, i])
        for v in V: 
            D_v = D[D[:, i] == v]
            values_Dv, counts_Dv = np.unique(D_v[:, -1], return_counts=True)
            P_v = counts_Dv/len(D_v)
            gini = 1 - np.sum(P_v**2)
            gini_avg += gini*np.sum(counts_Dv)/len(D[:, i])
        gini_values.append(gini_avg)
    return L[np.argmin(gini_values)]
    
def predict(node, instance, L):
    if node.data in [0, 1]:
        return node.data
    attrIndex = np.where(L == node.data)[0][0]
    if instance[attrIndex] == 0:
        return predict(node.left, instance, L)
    elif instance[attrIndex] == 1:
        return predict(node.middle, instance, L)
    else:
        return predict(node.right, instance, L)

def evaluate_accuracy(tree, data, L):
    correct = 0
    for instance in data.values:
        if predict(tree, instance, L) == instance[-1]:
            correct += 1
    return correct / len(data)

for i in range(100): 
    dataset = shuffle(dataset)
    L = dataset.columns.values
    trainData, testData = train_test_split(dataset, train_size=0.8, test_size=0.2)
    tree = decisionTree(trainData.values, L[:-1])
    train_accuracy.append(evaluate_accuracy(tree, trainData, L))
    test_accuracy.append(evaluate_accuracy(tree, testData, L))
    
# print(np.mean(train_accuracy), np.mean(test_accuracy))
# print(np.std(train_accuracy), np.std(test_accuracy))
# print(np.var(train_accuracy), np.var(test_accuracy))

#Q1
plt.subplot(2, 1, 1)
plt.hist(train_accuracy, bins=10, edgecolor='white')
plt.title('Training Accuracy Distribution with Gini Criteria')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')

#Q2
plt.subplot(2, 1, 2)
plt.hist(test_accuracy, bins=10, edgecolor='white')
plt.title('Testing Accuracy with Gini Criteria')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
