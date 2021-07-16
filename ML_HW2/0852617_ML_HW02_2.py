#!/usr/bin/env python
# coding: utf-8

# ### 0852617 曾鈺評 機器學習HW02_2
# ### Online Learning

# In[1]:


import numpy as np


# In[62]:


directory = "D:\\1091NCTU\Machine learning\\Homework\\"
rawdata = []
with open(directory+"testfile.txt") as file:
    for line in file:
        rawdata.append([int(i) for i in list(line.split("\n")[0])])


# In[96]:


def Factorial(number):
    result = 1
    for i in range(1, number+1, 1):
        result = result*i
    return result

def Combination(n,x):
    return Factorial(n) / (Factorial(x) * Factorial(n-x))

def Gamma(x):
    return Factorial(x-1)

def Beta(a, b):
    return (Gamma(a) * Gamma(b)) / Gamma(a+b)


# In[137]:


alpha = int(input("alpha = "))
beta = int(input("beta = "))


# In[138]:


for i in range(len(rawdata)):
    print("case %s:" %(i+1), end=' ')
    for j in rawdata[i]:
        print(j, end='')
    print("")
    n = len(rawdata[i])
    k = sum(rawdata[i])
    mu = k / n
    likelihood = Combination(n, k) * mu**k * (1-mu)**(n-k)
    print("Likelihood: %s" %likelihood)
    print("Beta prior: a= %s b=%s" %(alpha, beta))
    alpha = k + alpha
    beta = n - k + beta
    print("Beta posterior: a= %s b=%s\n" %(alpha, beta))


# In[139]:


alpha = int(input("alpha = "))
beta = int(input("beta = "))


# In[140]:


for i in range(len(rawdata)):
    print("case %s:" %(i+1), end=' ')
    for j in rawdata[i]:
        print(j, end='')
    print("")
    n = len(rawdata[i])
    k = sum(rawdata[i])
    mu = k / n
    likelihood = Combination(n, k) * mu**k * (1-mu)**(n-k)
    print("Likelihood: %s" %likelihood)
    print("Beta prior: a= %s b=%s" %(alpha, beta))
    alpha = k + alpha
    beta = n - k + beta
    print("Beta posterior: a= %s b=%s\n" %(alpha, beta))


# In[ ]:




