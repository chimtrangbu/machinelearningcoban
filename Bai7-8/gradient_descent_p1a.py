#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt


# In[2]:


def grad(x):
    return 2*x+ 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)


# In[3]:


(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))


# In[15]:


def gd_display(X):
    x0 = np.linspace(-5, 5, 50)
    y0 = cost(x0)
    
    x = np.array([X])
    y = np.array([[cost(e) for e in X]])
    
    plt.plot(x, y, 'ro')
    plt.plot(x0, y0, 'b-')
    plt.axis([-6, 6, -10, 30])
    plt.show()
    
gd_display(x1)
gd_display(x2)


# In[16]:


(x1, it1) = myGD1(.01, -5)
(x2, it2) = myGD1(.5, -5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
gd_display(x1)
gd_display(x2)


# In[ ]:




