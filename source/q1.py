import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

try:
    m = int(sys.argv[1])
except ValueError:
    print("please input m as integer")

data_path = "./"+sys.argv[2]

with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter = ' ')
    data = list(reader)
    data = np.array(data).astype(int)
    

ratios = []
m = 100
M = np.random.standard_normal((m,61200))

def dimension_reduction(data, m, examples, M):
    post_v = np.zeros([examples,m]).astype(float)
    #reduce dimentionality
    for i in range(1,examples+1):
        #fetch i-th post
        item_array = np.where(data[:,0] == i)[0]
        v_id = data[item_array,1] - 1
        v_count = data[item_array,2]
        
        post_v[i-1] = np.dot(M[:,v_id],v_count) 
    return post_v
 
post_v = dimension_reduction(data,m,1000,M)

#Euclidean distances
for i in range(1,1001):
    item_array_u = np.where(data[:,0] == i)[0]
    id_u = data[item_array_u,1]
    count_u = data[item_array_u,2]
    u = np.zeros([id_u.max()])
    for index, item in enumerate(id_u): 
        u[item-1] = count_u[index]
    
    u_bar = post_v[i-1]
    
    for j in range(i+1,1001):
        item_array_v = np.where(data[:,0] == j)[0]
        id_v = data[item_array_v,1]
        count_v = data[item_array_v,2]
        v = np.zeros([id_v.max()])
        dis_u_v = 0
        for index, item in enumerate(id_v): 
            v[item-1] = count_v[index]
        if len(v) > len(u):
            dis_u_v = np.sqrt(np.sum((v[:len(u)] - u)**2) + np.sum(v[len(u):]**2))
        elif len(u) > len(v):
            dis_u_v = np.sqrt(np.sum((u[:len(v)] - v)**2) + np.sum(u[len(v):]**2))
        else:
            dis_u_v = np.sqrt(np.sum((u - v)**2))
        
        if dis_u_v == 0:
            continue
        v_bar = post_v[j-1] 
        dis_u_v_bar = np.sqrt(np.sum((u_bar-v_bar)**2))
        ratios.append(dis_u_v_bar/dis_u_v)
        
plt.hist(ratios,bins=200)
plt.xlabel("Ratio values")
plt.ylabel("Frequency")
plt.show()
