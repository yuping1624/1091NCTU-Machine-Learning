#!/usr/bin/env python
# coding: utf-8

# ### 機器學習 HW3-1
# ### 統計所 0852617 曾鈺評

# ### 1. Random Data Generator
# ### a. Univariate gaussian data generator

# In[1]:


import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt


# In[2]:


m = int(input("mean m = "))
s = int(input("variance s = "))


# In[3]:


def NormalGenerator(m, s):
    u = random.uniform(0,1)
    v = random.uniform(0,1)
    z = math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*v)
    output = m + z*np.sqrt(s)
    return output


# In[4]:


output = NormalGenerator(m, s)
print("output = %s" %output)


# ### b. Polynomial basis linear model data generator

# In[5]:


def PolynomialLinearGenerator(n, a, w):
    w = np.array([float(i) for i in w.split(' ')])
    x = random.uniform(-1,1)
    x_polynomial = [x**i for i in range(n)]
    epsilon = NormalGenerator(0,1)*math.sqrt(a)
    #print(epsilon)
    y = sum([w_i*x_i for w_i, x_i in zip(w, x_polynomial)]) + epsilon
    return x, x_polynomial, y


# In[6]:


n = int(input("n = "))
a = int(input("a = "))
w = input("w = ")
x, x_polynomial, y = PolynomialLinearGenerator(n, a, w)
print("x = %s" % x)
print("y = %s" % y)


# ### 2. Sequential Estimator

# In[7]:


output = NormalGenerator(m, s)
Sum = output
SumSquare = output**2
n = 1
mean_previous = Sum / n
variance_previous = 0
threshold = 0.001
# mean_epsilon = abs(mean - mean_previous)
# var_epsilon = abs(variance - variance_previous)
mean_epsilon = threshold + 1 #abs(mean - mean_previous)
var_epsilon = threshold + 1 #abs(variance - variance_previous)


print("Data point source function: N(%.1f, %.1f)\n" %(m, s))
while (mean_epsilon > threshold) or (var_epsilon > threshold):
    added_data = NormalGenerator(m, s)
    n += 1
    Sum += added_data
    SumSquare += added_data**2
    mean = Sum / n
    variance = (SumSquare - Sum**2/n) / (n-1)
    
    mean_epsilon = abs(mean - mean_previous)
    var_epsilon = abs(variance - variance_previous)
#     mean_epsilon = abs(mean - m)
#     var_epsilon = abs(variance - s)
    
    print("Add data point: %s" % added_data)
    print("Mean = %s Variance = %s" % (mean, variance))
    
    mean_previous = mean
    variance_previous = variance


# ### 3. Baysian Linear regression

# In[8]:


def PrintPposterior(posterior_mean, posterior_variance, n):
    print("\nPosterior mean: ")
    for i in range(n):
        print("   %s" %str(posterior_mean[i])[1:-1])
    print("\nPosterior variance: ")
    for i in range(n):
        #for j in range(n):
        print("   %s" %str(posterior_variance[i])[1:-1])
        #print("\n")


# In[9]:


b = float(input("b = "))
n = int(input("n = "))
a = int(input("a = "))
w = input("w = ")


# In[10]:


X = []
Y = []
prior_mean = np.zeros(n).reshape(-1,1)
prior_variance = np.linalg.inv(b*np.eye(n))
precision = 1/a
#print(prior_variance)
epsilon = 0.001
count = 0

while (True):
    count += 1
    x, x_polynomial, y = PolynomialLinearGenerator(n, a, w)
    X.append(x)
    Y.append(y)
    print("Add data point: (%.4f, %.4f)" %(x,y))
    design_matrix = np.array([x**i for i in range(n)]).reshape(1,-1)
    #y = np.array(y).reshape(-1,1)
    posterior_variance = np.linalg.inv(precision*design_matrix.T@design_matrix + np.linalg.inv(prior_variance)) 
    posterior_mean = posterior_variance@(np.linalg.inv(prior_variance)@prior_mean + precision*design_matrix.T*y)
    PrintPposterior(posterior_mean, posterior_variance, n)
    #print(posterior_variance)
    predictive_variance = 1/precision + design_matrix@posterior_variance@design_matrix.T
    predictive_mean = design_matrix@posterior_mean
    print("\nPredictive distribution ~ N(%.5f, %.5f)" %(predictive_mean[0][0], predictive_variance[0][0]))
    print("--------------------------------------------------------")
    variance_epsilon = abs(predictive_variance[0][0] - a)
    if (count == 10):
        posterior_mean_ten = posterior_mean
        posterior_variance_ten = posterior_variance
    if (count == 50):
        posterior_mean_fifty = posterior_mean
        posterior_variance_fifty = posterior_variance
    if (variance_epsilon <= epsilon):
        break
    prior_mean = posterior_mean
    prior_variance = posterior_variance


# In[12]:


x = np.linspace(-2, 2, 30)
w = [float(w_i) for w_i in w.split(' ')]

fig = plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.title("Ground Truth")
plt.xlim(-2,2)
plt.ylim(-15,20)
ground_function = np.poly1d(np.flip(w))
y = ground_function(x)
y_plus = y + a
y_minus = y - a
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')

plt.subplot(2,2,2)
plt.title("Predict result")
plt.xlim(-2,2)
plt.ylim(-15,20)
predict_function = np.poly1d(np.flip(posterior_mean.flatten()))
for j in range(len(x)):
    design_matrix = np.array([x[j]**i for i in range(n)]).reshape(1,-1)
    y[j] = design_matrix@posterior_mean
    predict_variance = 1/precision + design_matrix@posterior_variance@design_matrix.T
    y_plus[j] = y[j] + predict_variance
    y_minus[j] = y[j] - predict_variance
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')
plt.scatter(X, Y)

plt.subplot(2,2,3)
plt.title("After 10 incomes")
plt.xlim(-2,2)
plt.ylim(-15,20)
predict_function = np.poly1d(np.flip(posterior_mean_ten.flatten()))
for j in range(len(x)):
    design_matrix = np.array([x[j]**i for i in range(n)]).reshape(1,-1)
    y[j] = design_matrix@posterior_mean_ten
    predict_variance = 1/precision + design_matrix@posterior_variance_ten@design_matrix.T
    y_plus[j] = y[j] + predict_variance
    y_minus[j] = y[j] - predict_variance
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')
plt.scatter(X[:10], Y[:10])


plt.subplot(2,2,4)
plt.title("After 50 incomes")
plt.xlim(-2,2)
plt.ylim(-15,20)
predict_function = np.poly1d(np.flip(posterior_mean_fifty.flatten()))
for j in range(len(x)):
    design_matrix = np.array([x[j]**i for i in range(n)]).reshape(1,-1)
    y[j] = design_matrix@posterior_mean_fifty
    predict_variance = 1/precision + design_matrix@posterior_variance_fifty@design_matrix.T
    y_plus[j] = y[j] + predict_variance
    y_minus[j] = y[j] - predict_variance
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')
plt.scatter(X[:50], Y[:50])
plt.show()


# In[13]:


b = float(input("b = "))
n = int(input("n = "))
a = int(input("a = "))
w = input("w = ")


# In[14]:


X = []
Y = []
prior_mean = np.zeros(n).reshape(-1,1)
prior_variance = np.linalg.inv(b*np.eye(n))
precision = 1/a
#print(prior_variance)
epsilon = 0.001
count = 0

while (True):
#for i in range(100):
    count += 1
    x, x_polynomial, y = PolynomialLinearGenerator(n, a, w)
    X.append(x)
    Y.append(y)
    print("Add data point: (%.4f, %.4f)" %(x,y))
    design_matrix = np.array([x**i for i in range(n)]).reshape(1,-1)
    #y = np.array(y).reshape(-1,1)
    posterior_variance = np.linalg.inv(precision*design_matrix.T@design_matrix + np.linalg.inv(prior_variance)) 
    posterior_mean = posterior_variance@(np.linalg.inv(prior_variance)@prior_mean + precision*design_matrix.T*y)
    PrintPposterior(posterior_mean, posterior_variance, n)
    #print(posterior_variance)
    predictive_variance = 1/precision + design_matrix@posterior_variance@design_matrix.T
    predictive_mean = design_matrix@posterior_mean
    print("\nPredictive distribution ~ N(%.5f, %.5f)" %(predictive_mean[0][0], predictive_variance[0][0]))
    print("--------------------------------------------------------")
    variance_epsilon = abs(predictive_variance[0][0] - a)
    if (count == 10):
        posterior_mean_ten = posterior_mean
        posterior_variance_ten = posterior_variance
    if (count == 50):
        posterior_mean_fifty = posterior_mean
        posterior_variance_fifty = posterior_variance
    if (variance_epsilon <= epsilon):
        break
    prior_mean = posterior_mean
    prior_variance = posterior_variance


# In[15]:


x = np.linspace(-2, 2, 30)
w = [float(w_i) for w_i in w.split(' ')]

fig = plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.title("Ground Truth")
plt.xlim(-2,2)
plt.ylim(-15,20)
ground_function = np.poly1d(np.flip(w))
y = ground_function(x)
y_plus = y + a
y_minus = y - a
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')

plt.subplot(2,2,2)
plt.title("Predict result")
plt.xlim(-2,2)
plt.ylim(-15,20)
predict_function = np.poly1d(np.flip(posterior_mean.flatten()))
for j in range(len(x)):
    design_matrix = np.array([x[j]**i for i in range(n)]).reshape(1,-1)
    y[j] = design_matrix@posterior_mean
    predict_variance = 1/precision + design_matrix@posterior_variance@design_matrix.T
    y_plus[j] = y[j] + predict_variance
    y_minus[j] = y[j] - predict_variance
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')
plt.scatter(X, Y)

plt.subplot(2,2,3)
plt.title("After 10 incomes")
plt.xlim(-2,2)
plt.ylim(-15,20)
predict_function = np.poly1d(np.flip(posterior_mean_ten.flatten()))
for j in range(len(x)):
    design_matrix = np.array([x[j]**i for i in range(n)]).reshape(1,-1)
    y[j] = design_matrix@posterior_mean_ten
    predict_variance = 1/precision + design_matrix@posterior_variance_ten@design_matrix.T
    y_plus[j] = y[j] + predict_variance
    y_minus[j] = y[j] - predict_variance
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')
plt.scatter(X[:10], Y[:10])


plt.subplot(2,2,4)
plt.title("After 50 incomes")
plt.xlim(-2,2)
plt.ylim(-15,20)
predict_function = np.poly1d(np.flip(posterior_mean_fifty.flatten()))
for j in range(len(x)):
    design_matrix = np.array([x[j]**i for i in range(n)]).reshape(1,-1)
    y[j] = design_matrix@posterior_mean_fifty
    predict_variance = 1/precision + design_matrix@posterior_variance_fifty@design_matrix.T
    y_plus[j] = y[j] + predict_variance
    y_minus[j] = y[j] - predict_variance
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')
plt.scatter(X[:50], Y[:50])
plt.show()


# In[16]:


b = float(input("b = "))
n = int(input("n = "))
a = int(input("a = "))
w = input("w = ")


# In[17]:


X = []
Y = []
prior_mean = np.zeros(n).reshape(-1,1)
prior_variance = np.linalg.inv(b*np.eye(n))
precision = 1/a
#print(prior_variance)
epsilon = 0.001
count = 0

while (True):
#for i in range(100):
    count += 1
    x, x_polynomial, y = PolynomialLinearGenerator(n, a, w)
    X.append(x)
    Y.append(y)
    print("Add data point: (%.4f, %.4f)" %(x,y))
    design_matrix = np.array([x**i for i in range(n)]).reshape(1,-1)
    #y = np.array(y).reshape(-1,1)
    posterior_variance = np.linalg.inv(precision*design_matrix.T@design_matrix + np.linalg.inv(prior_variance)) 
    posterior_mean = posterior_variance@(np.linalg.inv(prior_variance)@prior_mean + precision*design_matrix.T*y)
    PrintPposterior(posterior_mean, posterior_variance, n)
    #print(posterior_variance)
    predictive_variance = 1/precision + design_matrix@posterior_variance@design_matrix.T
    predictive_mean = design_matrix@posterior_mean
    print("\nPredictive distribution ~ N(%.5f, %.5f)" %(predictive_mean[0][0], predictive_variance[0][0]))
    print("--------------------------------------------------------")
    variance_epsilon = abs(predictive_variance[0][0] - a)
    if (count == 10):
        posterior_mean_ten = posterior_mean
        posterior_variance_ten = posterior_variance
    if (count == 50):
        posterior_mean_fifty = posterior_mean
        posterior_variance_fifty = posterior_variance
    if (variance_epsilon <= epsilon):
        break
    prior_mean = posterior_mean
    prior_variance = posterior_variance


# In[18]:


x = np.linspace(-2, 2, 30)
w = [float(w_i) for w_i in w.split(' ')]

fig = plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.title("Ground Truth")
plt.xlim(-2,2)
plt.ylim(-15,20)
ground_function = np.poly1d(np.flip(w))
y = ground_function(x)
y_plus = y + a
y_minus = y - a
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')

plt.subplot(2,2,2)
plt.title("Predict result")
plt.xlim(-2,2)
plt.ylim(-15,20)
predict_function = np.poly1d(np.flip(posterior_mean.flatten()))
for j in range(len(x)):
    design_matrix = np.array([x[j]**i for i in range(n)]).reshape(1,-1)
    y[j] = design_matrix@posterior_mean
    predict_variance = 1/precision + design_matrix@posterior_variance@design_matrix.T
    y_plus[j] = y[j] + predict_variance
    y_minus[j] = y[j] - predict_variance
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')
plt.scatter(X, Y)

plt.subplot(2,2,3)
plt.title("After 10 incomes")
plt.xlim(-2,2)
plt.ylim(-15,20)
predict_function = np.poly1d(np.flip(posterior_mean_ten.flatten()))
for j in range(len(x)):
    design_matrix = np.array([x[j]**i for i in range(n)]).reshape(1,-1)
    y[j] = design_matrix@posterior_mean_ten
    predict_variance = 1/precision + design_matrix@posterior_variance_ten@design_matrix.T
    y_plus[j] = y[j] + predict_variance
    y_minus[j] = y[j] - predict_variance
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')
plt.scatter(X[:10], Y[:10])


plt.subplot(2,2,4)
plt.title("After 50 incomes")
plt.xlim(-2,2)
plt.ylim(-15,20)
predict_function = np.poly1d(np.flip(posterior_mean_fifty.flatten()))
for j in range(len(x)):
    design_matrix = np.array([x[j]**i for i in range(n)]).reshape(1,-1)
    y[j] = design_matrix@posterior_mean_fifty
    predict_variance = 1/precision + design_matrix@posterior_variance_fifty@design_matrix.T
    y_plus[j] = y[j] + predict_variance
    y_minus[j] = y[j] - predict_variance
plt.plot(x, y, color='black')
plt.plot(x, y_plus, color='red')
plt.plot(x, y_minus, color='red')
plt.scatter(X[:50], Y[:50])
plt.show()


# In[ ]:




