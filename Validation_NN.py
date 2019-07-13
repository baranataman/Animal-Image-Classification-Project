from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random

# X_train ve Y_train olarak oku
X_train = np.round(np.genfromtxt('train_features.csv', delimiter=','),3)
Y_train = np.genfromtxt('train_labels.csv')

# sonra bunları birleştirip tek matrix oluştur ve shufflela
trainningset = np.c_[X_train, Y_train]
shuffledArray = random.sample(list(trainningset), len(trainningset))
shuffledArray = np.array(shuffledArray)

# bu shufflelanmış dataseti validation setlere böl
X_validation = shuffledArray[slice(20,55),:-1]
Y_validation = shuffledArray[slice(20,55),1000]

# sonra o shufflelanmışın validationunu sil
shuffledArray = np.delete(shuffledArray, slice( 20,55), axis=0)

# ve train seti de güncelle
X_train = shuffledArray[:, :-1] 
Y_train = shuffledArray[:, 1000]

# test seti oku
X_test = np.round(np.genfromtxt('test_features.csv', delimiter=','),3)
Y_test = np.genfromtxt('test_labels.csv')

# one hot encoding
encoder = OneHotEncoder()
Y_train_one_hot = encoder.fit_transform(np.expand_dims(Y_train,1)).toarray()
Y_validation_one_hot = encoder.fit_transform(np.expand_dims(Y_validation,1)).toarray()


def sig(z):  
    return 1/(1+np.exp(-z))

def softmax(z):  
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

def find_acc(found_labels,correct_labels):
    num = 0
    for i in range(0,found_labels.shape[0]):
        if found_labels[i] == correct_labels[i]:
            num = num + 1
    acc = float(num)/len(found_labels) * 100
    return acc
  
Cost_Function = []
out_neurons = 8
h_neurons = [75, 100, 125, 150, 175, 200]
learn_rate = [0.001, 0.003, 0.01, 0.03, 0.1]
Epoch = [250, 500, 750, 1000]

correct_n = 0
correct_l = 0
correct_e = 0
best_acc = 0

for neur in range(0,6):
    for lear in range(0,5):
        for epo in range(0,4):
            
            w_ih = np.random.rand(X_train.shape[1] ,h_neurons[neur]) # Weights between input and hidden layer
            w_ho = np.random.rand(h_neurons[neur],out_neurons) # Weights between hidden and output layer
            b_ih = np.random.randn(h_neurons[neur]) #Bias term between input and hidden layer
            b_ho = np.random.randn(out_neurons) #Bias term between hidden and output layer
            
            for i in range(Epoch[epo]):  
                
                # feedforward
                z0 = np.dot(X_train, w_ih) + b_ih
                h0 = sig(z0) # Hidden layer output
                z1 = np.dot(h0, w_ho) + b_ho
                a1 = softmax(z1) # Output layer output
            
                #Backpropagation
                grad_w_ho = np.dot(h0.T, (a1 - Y_train_one_hot)) #gradient for hidden layer output weights
                grad_b_ho = (a1 - Y_train_one_hot) #gradient for bias term from hidden layer 
                grad_w_ih = np.dot(X_train.T, sig(z0) *(1-sig(z0))*np.dot((a1 - Y_train_one_hot) , w_ho.T)) #gradient for input weights
                grad_b_ih = np.dot((a1 - Y_train_one_hot) , w_ho.T) * sig(z0) *(1-sig(z0)) #gradient for bias term for input layer
            
                #Updating the weights and bias terms
                w_ho =  w_ho - learn_rate[lear] * grad_w_ho
                w_ih = w_ih - learn_rate[lear] * grad_w_ih    
                b_ho = b_ho - learn_rate[lear] * grad_b_ho.sum(axis=0)
                b_ih = b_ih- learn_rate[lear] * grad_b_ih.sum(axis=0)
                
                Loss_Value = np.sum(-Y_train_one_hot * np.log(a1))
                Cost_Function.append(Loss_Value)
                print(Loss_Value)
                
            plt.plot(Cost_Function)
            plt.ylabel('Cost Function')
            plt.xlabel('Epoch')
            train_pred = np.argmax(a1, axis=1)
            train_acc = find_acc(train_pred,Y_train)
            print('Train Accuracy = ' + str(train_acc))


            # Forward propagating one with the new weights and test data
            z0_test = np.dot(X_validation, w_ih) + b_ih
            a0_test = sig(z0_test)
            z1_test = np.dot(a0_test, w_ho) + b_ho
            a1_test = softmax(z1_test)
            class_pred = np.argmax(a1_test, axis=1)
            Test_acc = find_acc(class_pred,Y_validation)
            print('Validation Accuracy with neurons '+ str(h_neurons[neur]) +', l_rate '+  str(learn_rate[lear]) + ', epoch ' +str(Epoch[epo])+ str(Test_acc))
            if ( best_acc < Test_acc):
                correct_n = h_neurons[neur]
                correct_l = learn_rate[lear]
                correct_e = Epoch[epo]
                best_acc = Test_acc
            #cm = confusion_matrix(Y_test,class_pred)
            plt.show()