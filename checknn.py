import keras
import tensorflow as tf
import h5py

from keras.models import load_model
import pandas as pd
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense


ar=[]


data=pd.read_csv('test.csv')
data=data.iloc[:2,:]

#Eliminating noise
data=data.drop(columns="Age")
data=data.drop(columns="Attrition")
data=data.drop(columns="DR")
data=data.drop(columns="DistanceFromHome")
data=data.drop(columns="EmployeeCount")
data=data.drop(columns="EmployeeID")
data=data.drop(columns="Gender")
data=data.drop(columns="MonthlyRate")
data=data.drop(columns="Over18")
data=data.drop(columns="StandardHours")
data=data.drop(columns="TrainingTimesLastYear")
data=data.drop(columns="YearsWithCurrManager")

def trav(s):
    if s=='Travel_Rarely':
        return 0.5
    elif s=='Non-Travel':
        return 0
    elif s=='Travel_Frequently':
        return 1
    else:
        return 0.5

def ov(s):
    if s=='Yes':
        return 1
    else:
        return 0
        
data['Travel']=data['Travel'].apply(trav)
data['OverTime']=data['OverTime'].apply(ov)
def vec1(dfc):
    l=['Human Resources', 'Research & Development', 'Sales']
    qq=len(l)
    ans=[]
    def f(n):
        p=[0 for i in range(qq)]
        for i in range(len(l)):
            print(n)
            if l[i]==n:
                p[i]=1
                break
        ans.append(p)
    dfc.apply(f)
    return np.asarray(ans)
def vec2(dfc):
    l=['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree']
    qq=len(l)
    ans=[]
    def f(n):
        p=[0 for i in range(qq)]
        for i in range(len(l)):
            if l[i]==n:
                p[i]=1
                break
        ans.append(p)
    dfc.apply(f)
    return np.asarray(ans)
def vec3(dfc):
    l=['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative']
    
    qq=len(l)
    ans=[]
    def f(n):
        p=[0 for i in range(qq)]
        for i in range(len(l)):
            if l[i]==n:
                p[i]=1
                break
        ans.append(p)
    dfc.apply(f)
    return np.asarray(ans)
def vec4(dfc):
    l=['Divorced', 'Married', 'Single']
    
    qq=len(l)
    ans=[]
    def f(n):
        p=[0 for i in range(qq)]
        for i in range(len(l)):
            if l[i]==n:
                p[i]=1
                break
        ans.append(p)
    dfc.apply(f)
    return np.asarray(ans)

q1=vec1(data['Dept'])
q2=vec2(data['EducationField'])
q3=vec3(data['Role'])
q4=vec4(data['Marital'])


#Dropping columns 
data=data.drop(columns="Dept")
data=data.drop(columns="EducationField")
data=data.drop(columns="Role")
data=data.drop(columns="Marital")
# data.fillna(0)
# print(q1)
# print('--------')
# print(q2)
# print('-----------')
# print(q3)
main=np.asarray(data)
tot=np.concatenate((main,q1,q2,q3,q4),axis=1)
arr=np.asarray(tot)
print(arr)
arr /=  arr.sum(axis=1)[:,np.newaxis]

model=load_model('JanPerformance.h5')

for i in arr:
    ansans=model.predict(arr)
    print(ansans)
