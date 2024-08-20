import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_csv('iris_dataset.csv', header=None)

training_stats = []
testing_stats = []

def train(data):
    features = data.iloc[:, :4].values
    label = data.iloc[:, 4].values

    maxVal = np.max(features, axis=0)
    minVal = np.min(features, axis=0)
    
    features = (features - minVal) / (maxVal - minVal)
    normalizedData = np.column_stack((features, label))
    
    return normalizedData, maxVal, minVal

def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def KNN_algorithm(trainData, unknown, k):
    features = trainData[:, :4]
    label = trainData[:, 4]

    distances = [euclidean_dist(unknown, feature) for feature in features]
    
    nearestIndices = np.argsort(distances)[:k]
    nearestNeighbors = label[nearestIndices]
    
    values, counts = np.unique(nearestNeighbors, return_counts=True)
    majority = values[np.argmax(counts)]

    return majority

for k in range(1, 52, 2):
    print(k)
    training_accuracy = []
    testing_accuracy = []
    
    for _ in range(20):
        dataset = shuffle(dataset)
        trainData, testData = train_test_split(dataset, train_size=0.8, test_size=0.2)
        normalized_trainData, maxVal, minVal = train(trainData)

        def evaluate_accuracy(data):
            correct = 0
            total = 0
            for i in range(len(data)):
                instance = data.iloc[i]
                normalized_instance = (instance[:4].values - minVal) / (maxVal - minVal)
                prediction = KNN_algorithm(normalized_trainData, normalized_instance, k)
                if prediction == instance[4]:
                    correct += 1
                total += 1
            return correct/total
        
        def evaluate_accuracy_without_norm(data):
            correct = 0
            total = 0
            for i in range(len(data)):
                instance = data.iloc[i]
                prediction = KNN_algorithm(trainData.values, instance[:4].values, k) #??
                if prediction == instance[4]:
                    correct += 1
                total += 1
            return correct/total
        
        # training_accuracy.append(evaluate_accuracy(trainData))
        # testing_accuracy.append(evaluate_accuracy(testData))

        training_accuracy.append(evaluate_accuracy_without_norm(trainData))
        testing_accuracy.append(evaluate_accuracy_without_norm(testData))

    training_stats.append((np.mean(training_accuracy), np.std(training_accuracy)))
    testing_stats.append((np.mean(testing_accuracy), np.std(testing_accuracy)))
        
training_stats = list(zip(*training_stats))
testing_stats = list(zip(*testing_stats))
print(np.mean(training_stats[0]), np.mean(testing_stats[0]))

#Q1
plt.subplot(2, 1, 1)
plt.errorbar(range(1, 52, 2), training_stats[0], yerr=training_stats[1], label='Training')
plt.xlabel('Value of k')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs k')
plt.legend()

#Q2
plt.subplot(2, 1, 2)
plt.errorbar(range(1, 52, 2), testing_stats[0], yerr=testing_stats[1], label='Testing')
plt.xlabel('Value of k')
plt.ylabel('Accuracy')
plt.title('Testing Accuracy vs k')
plt.legend()

plt.tight_layout()
plt.show()

