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
from sklearn.feature_selection import r_regression

sb.set_theme(style="darkgrid")

"""Step 1: Data In"""
df=pd.read_csv("Project_1_Data.csv")

#Converting Step column from Integar to 
#df['Step']=df['Step'].astype(float)



"""Step 2: Visalizing Data"""
    #Before visalizing, split data between train & test
    
    #Creates a new column that splits the data for Stratified Random Sampling
Split_1=StratifiedShuffleSplit (n_splits=1,test_size=0.2,random_state=42)
df["Split"] = pd.cut(df["Step"],
                          bins=[0, 2, 4, 6, np.inf],
                          labels=[1, 2, 3, 4])

    #Removing spit col that was added for SRS
for train_index, test_index in Split_1.split(df, df["Split"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
strat_df_train = strat_df_train.drop(columns=["Split"], axis = 1)
strat_df_test = strat_df_test.drop(columns=["Split"], axis = 1)    

    #Split data into X & Y
X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]

    #Creating histograms of split data
fig1, axs=plt.subplots(2,2,figsize=(10,10))
#fig1.set_dpi(1000)
strat_df_train.hist('X', ax=axs[0,0],bins=10)
axs[0,0].set_title('X Histogram')
strat_df_train.hist('Y', ax=axs[0,1],bins=10)
axs[0,1].set_title('Y Histogram')
strat_df_train.hist('Z', ax=axs[1,0],bins=10)
axs[1,0].set_title('Z Histogram')
strat_df_train.hist('Step', ax=axs[1,1],bins=10)
axs[1,1].set_title('Step Histogram')

    #3D plot creation
A=strat_df_train.to_numpy()
fig2 = plt.figure(figsize=(10,10))
ax=fig2.add_subplot(projection='3d')
fig2.set_dpi(1000)
x1=A[:,0]
y1=A[:,1]
z1=A[:,2]
ax.scatter(x1,y1,z1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')
plt.show(fig2)


"""Step 3: Correlation Analysis"""
    #Creating regressions of Features & Target 
plt.figure(figsize=(6, 6),dpi = 500)
fig3=sb.regplot(data=strat_df_train,x="Step",y="X")
fig3.set_title('X Regression')
plt.show()
plt.figure(figsize=(6, 6),dpi = 500)
fig4=sb.regplot(data=strat_df_train,x="Step",y="Y")
fig4.set_title('X Regression')
plt.show()
plt.figure(figsize=(6, 6),dpi = 500)
fig5=sb.regplot(data=strat_df_train,x="Step",y="Z")
fig5.set_title('Z Regression')
plt.show()


    #Calculating Pearson Correlation for X, Y, Z with Step
reg=r_regression(X_train,y_train)
print("Pearson Correlation Coefficients are",reg)


"""Scaling"""
my_scaler = StandardScaler()
my_scaler.fit(X_train.iloc[:,0:3])
scaled_data_train = my_scaler.transform(X_train.iloc[:,0:3])
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns=X_train.columns[0:3])
X_train = scaled_data_train_df.join(X_train.iloc[:,3:])

scaled_data_test = my_scaler.transform(X_test.iloc[:,0:3])
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns=X_test.columns[0:3])
X_test = scaled_data_test_df.join(X_test.iloc[:,3:])


#Print the Correlation Matrix
plt.figure(figsize=(6, 6),dpi = 500)
corr_matrix = (X_train).corr()
fig6=sb.heatmap(np.abs(corr_matrix),annot=True)
fig6.set_title('Correlation matrix')
#sb.pairplot(strat_df_train)

fig1, axs=plt.subplots(2,2,figsize=(10,10))
#fig1.set_dpi(1000)
scaled_data_train_df.hist('X', ax=axs[0,0],bins=10)
axs[0,0].set_title('X Histogram')
scaled_data_train_df.hist('Y', ax=axs[0,1],bins=10)
axs[0,1].set_title('Y Histogram')
scaled_data_train_df.hist('Z', ax=axs[1,0],bins=10)
axs[1,0].set_title('Z Histogram')


