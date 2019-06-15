import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import operator

try:
    k = int(sys.argv[1])
    m = int(sys.argv[2])
except ValueError:
    print("please input k and m as integer")

train_data_path = "./" + sys.argv[3]
train_label_path = "./" + sys.argv[4]
test_data_path = "./" + sys.argv[5]

with open(train_data_path, 'r') as f:
    reader = csv.reader(f, delimiter = ' ')
    train_data = list(reader)
    train_data = np.array(train_data).astype(int)
    
with open(train_label_path, 'r') as f:
    reader = csv.reader(f, delimiter = ' ')
    train_label = list(reader)
    train_label = np.array(train_label).astype(int)
    
with open(test_data_path, 'r') as f:
    reader = csv.reader(f, delimiter = ' ')
    test_data = list(reader)
    test_data = np.array(test_data).astype(int)

def dimension_reduction(data, m, examples, M):
    post_v = np.zeros([examples,m]).astype(float)
    for i in range(1,examples+1):
        #fetch i-th post
        item_array = np.where(data[:,0] == i)[0]
        v_id = data[item_array,1] - 1
        v_count = data[item_array,2]
        
        post_v[i-1] = np.dot(M[:,v_id],v_count) 
    return post_v
    
def get_neighbours(x_train, x_test_instance,k):
    distances = []
    neighbours = []
    for i in range(0, x_train.shape[0]):
        dist = np.sqrt(np.sum((x_train[i] - x_test_instance)**2))
        distances.append((i, dist))
        
    distances.sort(key=operator.itemgetter(1))
    
    for i in range(0,k): 
        neighbours.append(distances[i][0])
    return neighbours

def predictkNN(output, y_train):
    classVotes = {}
    for i in range(len(output)):
        if y_train[output[i],0] in classVotes:
            classVotes[y_train[output[i], 0]] += 1
        else:
            classVotes[y_train[output[i], 0]] = 1
        
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
        
# k-nearst neighbour classifier
def kNN_classifier(x_train, y_train, x_test, k):
    output_classes = np.zeros([x_test.shape[0],1]).astype(int)
    for i in range(0, x_test.shape[0]):
        output = get_neighbours(x_train, x_test[i], k)
        predicted = predictkNN(output, y_train)
        output_classes[i,0] = predicted
    return output_classes

M = np.random.standard_normal((m,61200))

train = dimension_reduction(train_data,m,1000, M)
test = dimension_reduction(test_data,m,100, M)

prediction = kNN_classifier(train, train_label,test, k)
with open("./predicted_label.csv", 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(prediction)
