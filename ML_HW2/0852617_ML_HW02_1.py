#!/usr/bin/env python
# coding: utf-8

# ### 0852617 曾鈺評 機器學習HW02_1

# In[1]:


import numpy as np
import gzip
import matplotlib.pyplot as plt
import math


# In[2]:


directory = "D:\\1091NCTU\Machine learning\\Homework\\"
IMAGE_SIZE = 28
NUM_TRAIN_IMAGES = 60000
NUM_TEST_IMAGES = 10000
NUM_CHANNELS = 1


# In[3]:


def ReadData(filename, num_char, num_images, path=directory, image_size=IMAGE_SIZE, num_channels=NUM_CHANNELS):
    with gzip.open(directory+filename) as file:
        file.read(num_char)
        buf = file.read(num_images * image_size * image_size * num_channels)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.int)
    return data


# In[4]:


def Prior(training_Y):
    prior = np.zeros(10)
    for i in range(10):
        prior[i] = np.sum(training_Y == i) / 60000
    return prior


# In[5]:


def DiscreteMode(training_X, training_Y, test_X):
    num_pixel_bin = np.zeros((10, 28*28, 32))
    num_pixel = np.zeros((10, 28*28))
    num = np.zeros(10)
    for i in range(10):
        num[i] = training_X[training_Y == i].shape[0]
        for j in range(28*28):
            for k in range(32):
                num_pixel_bin[i][j][k] = np.sum(training_X[training_Y == i, j] == k) / num[i]

    # 計算補0的probability
    pseudo_prob = num_pixel_bin[num_pixel_bin != 0].min()
    num_pixel_bin_new = num_pixel_bin.copy()
    num_pixel_bin_new[num_pixel_bin_new == 0] = pseudo_prob
    #0.0001483239394838327
    
    #計算10000筆資料在0~9分別的posterior (尚未加上prior)
    likelihood_discrete = np.zeros((10000,10))
    for n in range(10000):
        for i in range(10):
            likelihood_discrete[n][i] = np.log(num_pixel_bin_new[i][range(28*28),test_X[n]]).sum()
    return num_pixel_bin, likelihood_discrete


# In[6]:


def ContinuousMode(training_X, training_Y, test_X, epsilon=2000):
    mu = np.zeros((10, 784))
    variance = np.zeros((10, 784))
    gaussian_prob = np.zeros((10000,10,784))
    for i in range(10):
        for j in range(28*28):
            mu[i][j] = training_X[training_Y == i,j].mean()
            variance[i][j] = np.var(training_X[training_Y == i,j])
    variance[variance == 0] = epsilon

    for i in range(10):
        for j in range(28*28):
            gaussian_prob[:,i,j] = -0.5 * np.log(2*np.pi*variance[i][j]) - 0.5 * (test_X[:,j] - mu[i][j])**2 / variance[i][j]

    likelihood_gaussian = np.sum(gaussian_prob, axis=2)
    return mu, likelihood_gaussian


# In[7]:


def Posterior(likelihood, prior):
    posterior_temp = likelihood + np.log(prior)
    posterior = (posterior_temp.T / posterior_temp.sum(axis=1).T).T
    pred_Y = np.argmin(posterior, axis=1)
    return posterior, pred_Y


# In[8]:


def PrintPosterior(posterior, pred_Y, test_Y):
    for n in range(10000):
        print("Posterior (in log scale):")
        for i in range(10):
            print("%s: %s" %(i, posterior[n,i]))
        print("Prediction: %s, Ans: %s\n" %(pred_Y[n], test_Y[n]))


# In[9]:


def PrintImagination(images, toggle):
    print("Imagination of numbers in Bayesian classifier:")
    if toggle == 0:
        for i in range(10):
            print("%s:" %i)
            for j in range(28*28):
                if j % 28 == 27:
                    print((np.argmax(images[i], axis=1) >= 16)[j].astype(int))
                else:
                    print((np.argmax(images[i], axis=1) >= 16)[j].astype(int), end=' ')
            print("\n")
    if toggle == 1:
        for i in range(10):
            print("%s:" %i)
            for j in range(28*28):
                if j % 28 == 27:
                    print((images[i] >= 128)[j].astype(int))
                else:
                    print((images[i] >= 128)[j].astype(int), end=' ')
            print("\n")


# In[10]:


training_X = ReadData(filename="train-images-idx3-ubyte.gz", num_char=16, num_images=NUM_TRAIN_IMAGES)
training_Y = ReadData(filename="train-labels-idx1-ubyte.gz", num_char=8, num_images=NUM_TRAIN_IMAGES)
test_X = ReadData(filename="t10k-images-idx3-ubyte.gz", num_char=16, num_images=NUM_TRAIN_IMAGES)
test_Y = ReadData(filename="t10k-labels-idx1-ubyte.gz", num_char=8, num_images=NUM_TRAIN_IMAGES)

training_X = training_X.reshape(NUM_TRAIN_IMAGES, IMAGE_SIZE * IMAGE_SIZE)
test_X = test_X.reshape(NUM_TEST_IMAGES, IMAGE_SIZE * IMAGE_SIZE)


# In[16]:


def NaiveBayes_Discrete(training_X=training_X, training_Y=training_Y, test_X=test_X, text_Y=test_Y):
    prior = Prior(training_Y)
    training_X = (training_X / 8).astype(int)
    test_X = (test_X / 8).astype(int)
    num_pixel_bin, likelihood = DiscreteMode(training_X, training_Y, test_X)
    posterior, pred_Y = Posterior(likelihood, prior)
    PrintPosterior(posterior, pred_Y, test_Y)
    PrintImagination(num_pixel_bin, toggle)
    print("Error rate: %.4f" %(1 - np.mean(pred_Y == test_Y)))


# In[17]:


def NaiveBayes_Continuous(training_X=training_X, training_Y=training_Y, test_X=test_X, text_Y=test_Y):
    prior = Prior(training_Y)
    mu, likelihood = ContinuousMode(training_X, training_Y, test_X)
    posterior, pred_Y = Posterior(likelihood, prior)
    PrintPosterior(posterior, pred_Y, test_Y)
    PrintImagination(mu, toggle)
    print("Error rate: %.4f" %(1 - np.mean(pred_Y == test_Y)))


# In[22]:


toggle = int(input("Toggle = "))
prior = Prior(training_Y)
if toggle == 0:
    NaiveBayes_Discrete()
elif toggle == 1:
    NaiveBayes_Continuous()


# In[ ]:





# In[ ]:




