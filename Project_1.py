import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

sb.set_theme(style="darkgrid")

"""Step 1: Data In"""
df=pd.read_csv("Project_1_Data.csv")

#Converting Step column from Integar to 
#df['Step']=df['Step'].astype(float)

#Saving Pandas File as Array A
A=df.to_numpy()


"""Step 2: Visalizing Data"""
print(df.info())
fig1=df.hist('X')
fig2=df.hist('Y')
fig3=df.hist('Z')
fig4=df.hist('Step')

"""3D plot creation"""
fig5 = plt.figure()
ax=fig5.add_subplot(projection='3d')

x1=A[:,0]
y1=A[:,1]
z1=A[:,2]

ax.scatter(x1,y1,z1,s=A[:,3],c=A[:,3])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Location')
ax.legend()
plt.show(fig5)


"""Step 3: Correlation Analysis"""
sb.lmplot(data=df,x="Step",y="X")
sb.lmplot(data=df,x="Step",y="Y")
sb.lmplot(data=df,x="Step",y="Z")

sb.relplot(data=df,x='X',y='Y')



