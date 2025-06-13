import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
 
def sigmoid(x):
    result = 1/(1+math.e**(-x))
    return result
 
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(121)
x = np.linspace(-10,10)
y = sigmoid(x)
 
ax.spines['top'].set_color('none')  
ax.spines['right'].set_color('none')  
 
ax.xaxis.set_ticks_position('bottom')  
ax.spines['bottom'].set_position(('data',0))  
ax.set_xticks([-10,-5,0,5,10])  
ax.yaxis.set_ticks_position('left')  
ax.spines['left'].set_position(('data',0))  
ax.set_yticks([-1,-0.5,0.5,1])  
 
plt.plot(x,y,label = "Sigmoid",linestyle='-',color='blue')
plt.legend()

plt.savefig('sigmoid.png',dpi=200)

def relu(x):
    result=np.maximum(0,x)
    return result
 
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(121)
x = np.linspace(-10,10)
y = relu(x)
 
ax.spines['top'].set_color('none')  
ax.spines['right'].set_color('none')  
 
ax.xaxis.set_ticks_position('bottom')  
ax.spines['bottom'].set_position(('data',0))  
ax.set_xticks([-10,-5,0,5,10])  
ax.yaxis.set_ticks_position('left')  
ax.spines['left'].set_position(('data',0))  
ax.set_yticks([5,10])  
 
plt.plot(x,y,label = "ReLu",linestyle='-',color='blue')
plt.legend()

plt.savefig('relu.png',dpi=200)

def softmax(x):
    result = np.exp(x)/np.sum(np.exp(x), axis=0)
    return result
 
ax = fig.add_subplot(122)
x = np.linspace(-10,10)
y = softmax(x)
 
ax.spines['top'].set_color('none')  
ax.spines['right'].set_color('none')  
 
ax.xaxis.set_ticks_position('bottom')  
ax.spines['bottom'].set_position(('data',0)) 
 
ax.set_xticks([-10,-5,0,5,10])  
ax.yaxis.set_ticks_position('left')  
ax.spines['left'].set_position(('data',0))  
ax.set_yticks([-1,-0.5,0.5,1])  
 
plt.plot(x,y,label = "Softmax",linestyle='-',color='blue')
plt.legend()

plt.savefig('softmax.png',dpi=200)