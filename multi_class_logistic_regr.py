import numpy as np
import random

from numpy import genfromtxt
train_features = genfromtxt('train_features.csv', delimiter=',')
test_features =  genfromtxt('test_features.csv', delimiter=',')

#train_labels = genfromtxt('train_labels.csv', delimiter=',')
test_labels=  genfromtxt('test_labels.csv', delimiter=',')
train_labels=  genfromtxt('train_labels.csv', delimiter=',')
for i in range (0,160):
    if(train_labels[i] != 1):
        train_labels[i] = 0
trainningset = np.c_[train_features, train_labels]

iteration = np.array([1000, 1500, 2000, 2500, 3000, 3500])
learning = np.array([0.005, 0.01, 0.015, 0.02, 0.025])
errors = np.zeros((len(iteration), len(learning)))

for it in range(0,len(iteration)):
    for step in range(0, len(learning)):
        shuffledArray = random.sample(list(trainningset), len(trainningset))
        shuffledArray = np.array(shuffledArray)

        yedek = shuffledArray
        error = 0
        for i in range(1,6):
            validationSet = shuffledArray[slice( (i-1)*32,i*32)]
            #validationSet = list(np.float_(validationSet))
            shuffledArray = np.delete(shuffledArray, slice( (i-1)*32,i*32), axis=0)
            coefficients = np.ones((1001,1))
            for k in range(1,iteration[it]):
                z = coefficients[0] + np.dot(shuffledArray[:, :-1],coefficients[slice(1,1001)])
                h = 1/(1 + np.exp(-z))
                coefficients[0] = coefficients[0] + learning[step]*( sum(shuffledArray[:,slice(1000,1001)] - h) )
                result = np.dot(np.transpose(shuffledArray[:, :-1]), (shuffledArray[:,slice(1000,1001)] - h))
                coefficients[slice(1,1001)] = coefficients[slice(1,1001)] + learning[step]*result 
            
            z = coefficients[0] + np.dot(validationSet[:,slice(0,1000)],coefficients[slice(1,1001)])
            h = 1/(1 + np.exp(-z))
            error = error + sum(np.power((validationSet[:,slice(1000,1001)]-h),2 ))
            shuffledArray = yedek
        
        error = error / 5
        errors[it][step] = error


q = np.where(errors==errors.min())
row = q[0][0]
column = q[1][0]
iterate = iteration[row]
stepSize = learning[column]
shuffledArray = random.sample(list(trainningset), len(trainningset))
shuffledArray = np.array(shuffledArray)
coefficients = np.ones((1001,1))

for j in range(0, iterate):
    z = coefficients[0] + np.dot(shuffledArray[:, :-1],coefficients[slice(1,1001)])
    h = 1/(1 + np.exp(-z))
    #coefficients[0] = coefficients[0] + stepSize*( sum( (np.reshape(shuffledArray[:, 1000], (160,1)) - h) ) )
    result = np.dot(np.transpose(shuffledArray[:, :-1]), (shuffledArray[:, 1000] - h))
    result = np.transpose(result[:,0])

    t4 = np.zeros((1000,1))
    for i in range(0,1000):
        t4[i] = stepSize*result[i]
    coefficients[slice(1,1001)] = coefficients[slice(1,1001)] + t4

z = coefficients[0] + np.dot(test_features[slice(0,1000)],coefficients[slice(1,1001)])
h = 1/(1 + np.exp(-z))

#h = np.zeros((40,8))

#def fonksiyon(train_features, train_labels, test_features):


    #train_features = list(np.float_(train_features))
    #train_labels = list(np.float_(train_labels))
    #test_labels = list(np.float_(test_labels))
    
#trainningset = np.c_[train_features, train_labels]
#iteration = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000])
#learning = np.array([0.005, 0.01, 0.015, 0.02, 0.025])
#
#errors = np.zeros((len(iteration), len(learning)))
#
#for it in range(0,len(iteration)):
#    for step in range(0, len(learning)):
#        shuffledArray = random.sample(list(trainningset), len(trainningset))
##            shuffledArray = list(np.float_(shuffledArray))
#        yedek = shuffledArray
#        error = 0
#        for i in range(1,5):
#            validationSet = shuffledArray[slice( (i-1)*32,i*32)]
#            #validationSet = list(np.float_(validationSet))
#            shuffledArray = np.delete(shuffledArray, slice( (i-1)*32,i*32), axis=0)
#            coefficients = np.ones((1001,1))
#            for k in range(1,iteration[it]):
#                z = coefficients[0] + np.dot(shuffledArray[:, :-1],coefficients[slice(1,1001)])
#                h = 1/(1 + np.exp(-z))
#                coefficients[0] = coefficients[0] + learning[step]*( sum(shuffledArray[:,slice(1000,1001)] - h) )
#                result = np.dot(np.transpose(shuffledArray[:, :-1]), (shuffledArray[:,slice(1000,1001)] - h))
#                coefficients[slice(1,1001)] = coefficients[slice(1,1001)] + learning[step]*result 
#            validationSet = np.float_(validationSet)
#            z = coefficients[0] + np.dot(validationSet[:,slice(0,1000)],coefficients[slice(1,1001)])
#            h = 1/(1 + np.exp(-z))
#            error = error + sum(np.power((validationSet[:,slice(1000,1001)]-h),2 ))
#            shuffledArray = yedek
#        
#        error = error / 5
#        errors[it][step] = error
#    
#q = np.where(errors==errors.min())
#q = q[0][0]      
#e = errors[q]
#w = np.where(e==e.min())[0][0]
#test_features = list(np.float_(test_features))
#for j in range(0,iteration[w]):
#    z = coefficients[0] + np.dot(shuffledArray[:, :-1],coefficients[slice(1,1001)])
#    h = 1/(1 + np.exp(-z))
#    coefficients[0] = coefficients[0] + learning[q]*( sum(shuffledArray[:,slice(1000,1001)] - h) )
#    result = np.dot(np.transpose(shuffledArray[:, :-1]), (shuffledArray[:,slice(1000,1001)] - h))
#    coefficients[slice(1,1001)] = coefficients[slice(1,1001)] + learning[step]*result 
#z = coefficients[0] + np.dot(test_features[slice(0,1000)],coefficients[slice(1,1001)])
#h = 1/(1 + np.exp(-z))
#    return h

