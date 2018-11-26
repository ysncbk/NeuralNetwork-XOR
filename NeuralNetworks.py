"""HW11 Machine Learning, MLP
Diego Lainfiesta and Yasin Cibuk"""


#Part A-----------------XOR function---------------------------------------------------------------------------

#Importing classes
import numpy as np
import matplotlib.pyplot as plt

#Defining important functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def square(z):
    return z**2

def cost(predicted,truth):
    
    return (predicted - truth)

#Setting the seed
    
np.random.seed(24)

#Initializing the weights and bias
w1 = np.random.uniform(-0.1, 0.1, size = (2,3))
w2 = np.random.uniform(-0.1, 0.1, size = (1,3))

a0 = np.array([[0,1,0,1],[0,1,1,0],[1,1,1,1]])
    
#Bias in hidden layer
bias1 = np.ones((1,4))

#Observed output
y = np.array([[0,0,1,1]])

list_w1 =[]
list_w2 =[]

epoch_list = []
lr = 0.15

loss_if_list =[]
loss_wo_if_list = []

for epoch in range (12000):
   
    epoch_list.append(epoch)
        
    #Input layer
   
    #Input of hidden layer
    z1 = w1.dot(a0)
        
    #Output of hidden layer
    a1 = sigmoid(z1) 
    a1b = np.concatenate((a1, bias1), axis = 0)
    
    #Input of output layer
    z2 = w2.dot(a1b)
    
    a2 = z2
    
    y2 = (a2 > 0.5).astype(int)
    
    print(epoch)
    
    difference_if_statement = y2-y
    loss_if =  np.apply_along_axis(square, 0, difference_if_statement)
    loss_if = np.sum(loss_if)/loss_if.shape[1]
    print(loss_if)
    loss_if_list.append(loss_if)
    
    difference_wo_if_statement = a2-y
    loss_wo_if =  np.apply_along_axis(square, 0, difference_wo_if_statement)
    loss_wo_if = np.sum(loss_wo_if)/loss_wo_if.shape[1]
    print(loss_wo_if)
    loss_wo_if_list.append(loss_wo_if) 
    
    #Backpropagation algorithm

    delta2 = 2*(a2 - y) #* a2 * (1-a2)
    
    dw_out = delta2.dot(a1b.T)/a1.shape[1]
    
    w2 = w2 - lr * dw_out
    list_w2.append(w2)
    dw_1 = (w2[:,0:2].T.dot(delta2) *a1*(1-a1)).dot(a0.T)/a0.shape[1]
    
    w1 = w1 - lr * dw_1
    list_w1.append(w1)
    
plt.plot(loss_if_list)
plt.plot(loss_wo_if_list)

len(loss_if_list)
loss_if_list[11999]

len(loss_wo_if_list)
loss_wo_if_list[11999]

loss_if_list[8564]
loss_wo_if_list[8564]

loss_wo_if_list[11999]

#Plotting of MSE with if >=0.5, then 1, else 0

fig1, ax1 = plt.subplots()
ax1.plot(loss_if_list, 'r----', label = "Mean Squared error from if Output")
ax1.set(xlabel='Epoc', ylabel='Mean Squared error from if Output', title = 'MSE vs. Epoch')
plt.show()


#Plotting of MSE without if >=0.5, then 1, else 0
    
fig2, ax1 = plt.subplots()
ax1.plot(loss_wo_if_list, 'b----', label = "Mean Squared error without if Output")
ax1.set(xlabel='Epoc', ylabel='Mean Squared error', title = 'MSE vs. Epoch')
plt.show()

# Part B--------------Continuos XOR function------------------------------------------------------------------- 

#Importing classes
import numpy as np
import matplotlib.pyplot as plt

#Defining important functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def square(z):
    return z**2

def cost(predicted ,truth):
    
    return (predicted - truth)

# Inputs for XOR
# number of samples of each class

np.random.seed(45)

first  = []
second = []
third  = []
fourth = []

for i in range (300):
    x , y = np.random.random() , np.random.random()
    a = [x,y]
    first.append(a)
    x , y = np.random.random() , np.random.random()
    a=[-x,-y]
    second.append(a)
    x , y = np.random.random() , np.random.random()
    a=[x,-y]
    third.append(a)
    x , y = np.random.random() , np.random.random()
    a=[-x,y]
    fourth.append(a)

#Plotting vecors

first = np.asarray(first)
x, y = first.T
plt.scatter(x,y)
second = np.asarray(second)
x, y = second.T
plt.scatter(x,y)
third = np.asarray(third)
x, y = third.T
plt.scatter(x,y)
fourth = np.asarray(fourth)
x, y = fourth.T
plt.scatter(x,y)

#Creating input matrix

a0 = np.concatenate((first, second, third, fourth), axis=0)
bias0 = np.ones((1,1200)).T
a0=np.concatenate((a0,bias0), axis=1)
a0 = a0.T

#Creating output matrix

y1 = np.zeros((1,600)).T
y2 = np.ones((1,600)).T
Y =  np.concatenate((y1, y2))
Y = Y.T

#Random initialisation of weights


w1 = np.random.uniform(-1, 1 ,size = (6, 3))
w2 = np.random.uniform(-2, 2 ,size = (1, 7))

#Bias hidden layer

bias1 = np.ones((1, 1200))

#Lists

trn_er= []
trn_er_if= []
w2_list = []
w1_list = []
epoch_list = []
lr_list = []

lr = 0.15

for epoch in range(7300):

    # Forward propagation.
    # Inside the perceptron
    
    print(epoch)
    epoch_list.append(epoch)
    lr_list.append(lr)
    
    z1 = np.dot(w1, a0)
    a1 = sigmoid(z1)
    
    a1b = np.concatenate((a1, bias1), axis = 0)
    
    z2 = np.dot(w2, a1b)
    
    a2 = z2
    #a2 = sigmoid(z2)
    
    a2_if = (a2 > 0.5).astype(int)
    
    # Back propagation (Y -> output)    
    
    # loss function
    
    ## Without if statement
    output_error = cost(a2 , Y)
    
    MSE = np.sum(np.apply_along_axis(square, 0, output_error))/output_error.shape[1]
    print(MSE)
    
    trn_er.append(MSE)
    
    ## With if statement
    
    output_error_if = cost(a2_if , Y)
    
    MSE_if = np.sum(np.apply_along_axis(square, 0, output_error_if))/output_error_if.shape[1]
    print(MSE_if)
    
    trn_er_if.append(MSE_if)
        
    #Sigma in the output
    
    delta2 = 2*(output_error) #* a2*(1-a2)
    
    #Delta dw2

    dw2 = delta2.dot(a1b.T)/delta2.shape[1]
    
    w2 = w2 - lr * dw2
    
    w2_list.append(w2)
    
    #Delta dw1

    dw1 = (w2[0:1,0:6].T.dot(delta2) * (a1*(1-a1))).dot(a0.T)/a0.shape[1]
    
    w1 = w1 - lr * dw1
    
    w1_list.append(w1)

#Plotting of MSE without if >=0.5, then 1, else 0
    
fig2, ax1 = plt.subplots()
ax1.plot(trn_er, 'b----', label = "Mean Squared error without if Output")
ax1.set(xlabel='Epoc', ylabel='Mean Squared error', title = 'MSE vs. Epoch')
plt.show()

#Plotting of MSE with if >=0.5, then 1, else 0

fig1, ax1 = plt.subplots()
ax1.plot(trn_er_if, 'r----', label = "Mean Squared error from if Output")
ax1.set(xlabel='Epoc', ylabel='Mean Squared error from if Output', title = 'MSE vs. Epoch')
plt.show()

#Statistics
min(trn_er_if)
trn_er_if.index(min(trn_er_if))

min(trn_er)
trn_er.index(min(trn_er))

#Plotting decision boundaries----------------------------------------------------------------------------------

np.random.seed(85)

#Creating mesh as input (equally spaced input values)

xx, yy = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))

xx.shape
yy.shape

#Creating from the mesh the input values

x = xx.reshape(1,1000000)
y = yy.reshape(1,1000000)

#Creating input matrix, with x, y and bias

a0_test = np.concatenate((x, y), axis=0) 
bias0_test = np.ones((1,1000000))

a0_test = np.concatenate((a0_test, bias0_test), axis=0)

#Bias vector for hidden layer

bias1_test = np.ones((1,1000000))

#Computing output with the previously trained MLP

z1_test = np.dot(w1, a0_test)
a1_test = sigmoid(z1_test)

a1b_test = np.concatenate((a1_test, bias1_test), axis = 0)
    
z2_test = np.dot(w2, a1b_test)

#Output from MLP (without and with if)
    
a2_test = z2_test
    
a2_if_test = (a2_test > 0.5).astype(int)

#Reshaping the output, to have it as a grid

a2_test = np.reshape(a2_test, (1000, 1000))
a2_if_test = np.reshape(a2_if_test, (1000,1000)) 

#Creating the 2-Dim Graph

from scipy.interpolate import griddata

#Adding the mesh of test data points and boundaries
plt.contour(xx, yy, a2_if_test, 15, linewidths=0.5, colors='k')
plt.contourf(xx, yy, a2_if_test, 15, cmap= plt.cm.jet)

#Adding the color bar
plt.colorbar(ticks = range(3), label = 'value')
#plt.clim(-0.5, 2)

#Adding the training data points in different colors for each category
plt.scatter(a0[0,0:299], a0[1,0:299], marker ='x', c='black', s=2)
plt.scatter(a0[0,300:599], a0[1,300:599], marker ='x', c='white', s=2)
plt.scatter(a0[0,600:899], a0[1,600:899], marker ='x', c='yellow', s=2)
plt.scatter(a0[0,900:1199], a0[1,900:1199], marker ='x', c='magenta', s=2)

#General formatting of the graph window
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('2-Dim for continuos XOR plot')
plt.show()

#Plotting Optimization Path-------------------------------------------------------------------------------------

len(w1_list)
len(w2_list)

a = np.zeros(shape=(1,25))

for i in range (7300):
    b = np.concatenate((w1_list[i].reshape(1,18), w2_list[i]), axis = 1)
    a = np.concatenate((a,b), axis = 0)
    
a = a[1:7301,0:25]

a.shape
means = a.mean(axis = 0).reshape(1,25)
a_centered = a - means

np.mean(a_centered[:,0:1])
np.std(a_centered[:,0:1])


norm_a = np.linalg.norm(a_centered, axis = 0)
norm_a.shape
norm_a = norm_a.reshape(1, 25)

a_centered_scaled = a_centered/norm_a

np.mean(a_centered_scaled[:,0:1])
np.std(a_centered_scaled[:,0:1])

A = 1/7300*np.dot(a_centered_scaled.T,a_centered_scaled)

u, s, vh = np.linalg.svd(A, full_matrices = True)

Importance_train = np.matrix.round(s/sum(s)*100,decimals=2)
Importance_train

Loadings = np.dot(a_centered_scaled, u[:,0:2])
Loadings.shape

from mpl_toolkits import mplot3d


fig = plt.figure()
fig.set_size_inches(12, 7)
ax = plt.axes(projection='3d')


zdata = trn_er_if
xdata = Loadings[:,0:1]
ydata = Loadings[:,1:2]
ax.scatter3D(xdata, ydata, zdata, c= "red")

ax.set_title('Optimization path-Dimension reduction from 25D to 2D')
ax.set_xlabel('U1', fontsize =10)
ax.set_ylabel('U2', fontsize =10)
ax.set_zlabel('Loss', fontsize =10)
ax.tick_params(labelsize= 10)





