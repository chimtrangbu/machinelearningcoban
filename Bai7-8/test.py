#!/usr/bin/env python
# coding: utf-8

# In[9]:


from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt


# In[10]:


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


# In[11]:


(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))


# In[21]:


def gd_display(X):
    fig, ax = plt.subplots()

    x0 = np.linspace(-5, 5, 50)
    y0 = cost(x0)
    
    x = np.array([X])
    y = np.array([[cost(e) for e in X]])
    ax.plot(x0, y0, 'b-')
    ax.axis([-6, 6, -10, 30])
    
    for i in range(len(x[0])):
        ax.plot(x[0][i], y[0][i], 'ro')
        plt.pause(2)
#    plt.show()
    
gd_display(x1)
gd_display(x2)

