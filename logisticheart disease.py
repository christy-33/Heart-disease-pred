#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib as plt
import pandas as pd


# In[4]:


data=pd.read_csv("C:\\Users\\CHRISTY HARSHITHA\\Downloads\\framingham_heart_disease.csv")


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.isnull().sum()


# In[9]:


"""
bpmeds dont disturb
edc


cigsperday(avg)
totchol avg
bmi avg
heart rate avg
glucose avg

"""


# In[10]:


data['totChol'] = data['totChol'].fillna(value=data['totChol'].mean())
data['cigsPerDay']=data['cigsPerDay'].fillna(value=data['cigsPerDay'].mean())
data['BMI']=data['BMI'].fillna(value=data['BMI'].mean())
data['heartRate']=data['heartRate'].fillna(value=data['heartRate'].mean())
data['glucose']=data['glucose'].fillna(value=data['glucose'].mean())


# In[11]:


data['BPMeds'] = data['BPMeds'].fillna(1)
data['education'] = data['education'].fillna(1)


# In[12]:


data.isnull().sum()


# In[13]:


data=data.sample(frac=1)


# In[14]:


data


# In[15]:


ratio=0.75
rows=data.shape[0]
train_size=int(rows*ratio)
train_set=data[0:train_size]
test_set=data[train_size:]


# In[16]:


print("training set")
print(train_set)


# In[17]:


x=train_set.drop(["TenYearCHD"],axis=1).values
y=train_set["TenYearCHD"].values


# In[18]:


x_test=test_set.drop(["TenYearCHD"],axis=1).values
y_test=test_set["TenYearCHD"].values


# In[19]:


print(x)


# In[20]:


print(x_test)


# In[21]:


print(x_test.shape)


# In[22]:


print(y)


# In[23]:


print(x.shape)


# In[24]:


print(y.shape)


# In[25]:


def sigmoid(z):
    k=1/(1+np.exp(-z))
    
    return k


# In[26]:


def cost_calc(x,y,w,b,*argv):
    m,n=x.shape
    cost=0
    for i in range(m):
        z=np.dot(x[i],w)+b
        f_wb=sigmoid(z)
        cost+=(-y[i]*np.log(f_wb))-((1-y[i])*np.log(1-f_wb))
        
    return cost/m
        
    
    
    


# In[27]:


#TEST SIGMOID
print(sigmoid(0))


# In[28]:


m=x.shape[0]
print(m)


# In[29]:


#test cost_calc
m,n=x.shape
print(cost_calc(x,y,np.zeros(n),1))


# In[30]:


def grad_calc(x,y,w,b):
    """
    dj_dw,dj_db>>>to update w and b
    """
    
    m,n=x.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
        z=np.dot(x[i],w)+b
        f_wb_i=sigmoid(z)
        err=f_wb_i-y[i]
        dj_db+=err
        for j in range(n):
            dj_dw[j]+=(err)*x[i,j]
    dj_db=dj_db/m
    dj_dw=dj_dw/m
    
    return dj_dw,dj_db
        


# In[31]:


def grad_descent(w_in,b_in,x,y,alpha,iter_count):
    j_his=[]
    
    m,n=x.shape
    for i in range(iter_count):
        dj_dw,dj_db=grad_calc(x,y,w_in,b_in)
        w_in=w_in-(alpha*dj_dw)
        b_in=b_in-(alpha*dj_db)
        
        if i<100000:
            cost=cost_calc(x,y,w_in,b_in)
            j_his.append(cost)
            
        if i % (iter_count // 10) == 0:
            print(f"iteration={i:4}   cost={float(j_his[-1]):8.5f}")
    return w_in,b_in,j_his


# In[57]:


np.random.seed(1)
w_in = np.zeros(n)
b_in = -8

# Some gradient descent settings
iterations = 1000
alpha = 1e-7

w,b, J_history = grad_descent(w_in,b_in,,y,alpha,iterations)


# In[58]:


w1=w
b1=b
print(w)
print(b)


# In[61]:


w1,b1, J_history = grad_descent(w_in,b_in,x,y,1e-6,10000)


# In[62]:


print(w1)
print(b)


# In[63]:


w2,b2, J_history = grad_descent(w1,b1,x,y,1e-6,10000)


# In[65]:


print(w2)
print(b2)


# In[67]:


w3,b3, J_history = grad_descent(w2,b2,x,y,1e-5,10000)
print(f"w3={w3}")
print(f"b2={b3}")


# In[68]:


w4,b4, J_history = grad_descent(w3,b3,x,y,1e-4,10000)
print(f"w3={w4}")
print(f"b2={b4}")


# In[70]:


w5,b5, J_history = grad_descent(w4,b4,x,y,1e-4,10000)
print(f"w5={w5}")
print(f"b5={b5}")


# In[32]:


w5=[ 0.02288544,  0.06100065,  0.0006614,  -0.00375587 , 0.02895242,  0.0028942,
  0.00277806,  0.01110768,  0.00308168,  0.00097438  ,0.01811674 ,-0.00174959,
  0.0076226,  -0.00802602 , 0.00873743]


# In[48]:


def predict(x,w,b):
    p=[]
    m,n=x.shape
    for i in range(m):
        z_i=np.dot(x[i],w)+b
        f_wb_i=sigmoid(z_i)
        if f_wb_i>=0.7:
            p.append(1)
        else:
            p.append(0)
    return p
    
            


# In[49]:


w5=[ 0.02288544,  0.06100065,  0.0006614,  -0.00375587,  0.02895242,  0.0028942,
  0.00277806,  0.01110768,  0.00308168,  0.00097438,  0.01811674, -0.00174959,
  0.0076226,  -0.00802602,  0.00873743]
b5=-8.000686871086703
p=predict(x_test,w5,b5)
print(p)
print(len(p))


# In[50]:


print('Train Accuracy: %f'%(np.mean(p == y_test) * 100))


# In[ ]:





# In[ ]:




