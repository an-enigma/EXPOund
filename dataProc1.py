import numpy as np
import pandas as pd

data=pd.read_csv('employee_data.csv')

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

#One hot vectorisation
def vec(dfc):
    l=sorted(list(set(list(dfc))))
    print(l)
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

def ov(s):
    if s=='Yes':
        return 1
    else:
        return 0
data['Travel']=data['Travel'].apply(trav)
data['OverTime']=data['OverTime'].apply(ov)

q1=vec(data['Dept'])
q2=vec(data['EducationField'])
q3=vec(data['Role'])
q4=vec(data['Marital'])


#Dropping columns 
data=data.drop(columns="Dept")
data=data.drop(columns="EducationField")
data=data.drop(columns="Role")
data=data.drop(columns="Marital")

data.fillna(0)

data['PerformanceRating']=data['PerformanceRating'].apply(lambda x : int(x)-3)
label=data['PerformanceRating']
qm=np.asarray(label)
trainlab=qm[:1200]
testlab=qm[1200:]
np.save('trainlabel',trainlab)
np.save('testlabel',testlab)
print(trainlab.shape)
print(testlab.shape)
#dropping the answer column
data=data.drop(columns="PerformanceRating")


main=np.asarray(data)
tot=np.concatenate((main,q1,q2,q3,q4),axis=1)
train=tot[:1200,:]
test=tot[1200:,:]
print(train.shape)
print(test.shape)

np.save('traindata',train)
np.save('testdata',test)
