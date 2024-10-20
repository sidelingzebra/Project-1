import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import r_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

from matplotlib import colormaps
from pandas.plotting import scatter_matrix
from sklearn.ensemble import StackingClassifier
import joblib



sb.set_theme(style="darkgrid")

"""Step 1: Data In"""
df=pd.read_csv("Project_1_Data.csv")

"""Step 2: Visalizing Data"""

    #Before visalizing, split data between train & test
    
    #Creates a new column that splits the data for Stratified Random Sampling
df["Split"] = pd.cut(df["Step"],
                          bins=[0, 2, 4, 6, np.inf],
                          labels=[1, 2, 3, 4])
Split_1=StratifiedShuffleSplit (n_splits=1,test_size=0.3,random_state=24)

    #Removing split col that was added for SRS
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
fig2 = plt.figure(figsize=(8,8))
ax=fig2.add_subplot(projection='3d')
fig2.set_dpi(1000)
x1=A[:,0]
y1=A[:,1]
z1=A[:,2]
fig2=ax.scatter(x1,y1,z1,c=A[:,3],cmap=plt.cm.viridis)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')
ax.legend(*fig2.legend_elements(),ncol=6,
          bbox_to_anchor=(1, 1) )
plt.show(fig2)


"""Step 3: Correlation Analysis"""
    #Creating regressions of Features & Target 
    
plt.figure(figsize=(6, 6),dpi = 500)
fig3=sb.regplot(data=strat_df_train,x="Step",y="X")
fig3.set_title('X-Coordinate vs. Maintenance Step')
plt.show(fig3)
plt.figure(figsize=(6, 6),dpi = 500)
fig4=sb.regplot(data=strat_df_train,x="Step",y="Y")
fig4.set_title('Y-Coordinate vs. Maintenance Step')
plt.show(fig4)
plt.figure(figsize=(6, 6),dpi = 500)
fig5=sb.regplot(data=strat_df_train,x="Step",y="Z")
fig5.set_title('Z-Coordinate vs. Maintenance Step')
plt.show(fig5)



    #Calculating Pearson Correlation for X, Y, Z with Step
reg=r_regression(X_train,y_train)
print("Pearson Correlation Coefficients are",reg)


"""Scaling"""
my_scaler = StandardScaler()
my_scaler.fit(X_train.iloc[:,0:3])
scaled_data_train = my_scaler.transform(X_train.iloc[:,0:3])
scaled_data_train_df = pd.DataFrame(scaled_data_train, 
                                    columns=X_train.columns[0:3])
X_train = scaled_data_train_df.join(X_train.iloc[:,3:])


scaled_data_test = my_scaler.transform(X_test.iloc[:,0:3])
scaled_data_test_df = pd.DataFrame(scaled_data_test, 
                                    columns=X_test.columns[0:3])
X_test = scaled_data_test_df.join(X_test.iloc[:,3:])


#Print the Correlation Matrix
plt.figure(figsize=(6, 6),dpi = 500)
corr_matrix = (strat_df_train).corr()
fig6=sb.heatmap(np.abs(corr_matrix),annot=True)
fig6.set_title('Correlation matrix')
#sb.pairplot(strat_df_train)




"""ML Model 1 - Linear Regression"""
linear_reg = LogisticRegression()
param_grid_lr = {}
grid_search_lr = GridSearchCV(linear_reg, param_grid_lr, cv=5, 
                              scoring='neg_mean_absolute_error', 
                              n_jobs=-1)

grid_search_lr.fit(X_train, y_train)

best_model_lr = grid_search_lr.best_estimator_
print("\nBest Linear Regression Model:", best_model_lr)

lr_prediction = grid_search_lr.predict(X_test)
print("\nLogistic Regression Accuracy Score:", accuracy_score(lr_prediction, y_test))

#Logistic Regression Confusion Matrix
fig, ax=plt.subplots(1,1,figsize=(6, 6), dpi = 500)
ConfusionMatrixDisplay.from_predictions(lr_prediction, 
                                            y_test,cmap=plt.cm.BuPu,ax=ax)
ax.set_title('Logistic Regression Confusion Matrix')
plt.show()


# print(confusion_matrix(lr_prediction, y_test))
print("\n",classification_report(lr_prediction, y_test))





"""ML Model 2 - Random Forest"""

forest_reg = RandomForestClassifier()

param_grid_rf = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search_rf = GridSearchCV(forest_reg, param_grid_rf, cv=5,
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1)

grid_search_rf.fit(X_train,y_train)


rf_prediction = grid_search_rf.predict(X_test)
print("\nRandom Forest Accuracy Score:", accuracy_score(rf_prediction, y_test))


#Random Forest Confusion Matrix
fig, ax=plt.subplots(1,1,figsize=(6, 6), dpi = 500)
ConfusionMatrixDisplay.from_predictions(rf_prediction, 
                                            y_test,cmap=plt.cm.BuPu,ax=ax)
ax.set_title('Random Forest Confusion Matrix')
plt.show()

# print(confusion_matrix(rf_prediction, y_test))
print("\n",classification_report(rf_prediction, y_test))



"""ML Model 3 - Decision Tree"""
decision_tree = DecisionTreeClassifier(random_state=24)

param_grid_dt = {
    'max_depth': [10, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = RandomizedSearchCV(decision_tree, param_grid_dt, cv=5,
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1)

grid_search_dt.fit(X_train,y_train)



dt_prediction = grid_search_dt.predict(X_test)
print("\nDecision Tree Accuracy Score:", accuracy_score(dt_prediction, y_test))

#Decision Tree Confusion Matrix
fig, ax=plt.subplots(1,1,figsize=(6, 6), dpi = 500)
ConfusionMatrixDisplay.from_predictions(dt_prediction, 
                                            y_test,cmap=plt.cm.BuPu,ax=ax)
ax.set_title('Decision Tree Confusion Matrix')
plt.show()


print("\n",classification_report(dt_prediction, y_test))





"""ML Model 4 - Support Vector Machine"""
svc = SVC()
param_grid_svc = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}

grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5, 
                               scoring='neg_mean_absolute_error', 
                               n_jobs=-1)

grid_search_svc.fit(X_train, y_train)


svc_prediction = grid_search_svc.predict(X_test)
print("\nSupport Vector Machine Accuracy Score:", accuracy_score(svc_prediction, 
                                                                 y_test))

#Support Vector Machine Confusion Matrix
fig, ax=plt.subplots(1,1,figsize=(6, 6), dpi = 500)
ConfusionMatrixDisplay.from_predictions(svc_prediction, 
                                            y_test,cmap=plt.cm.BuPu,ax=ax)
ax.set_title('Support Vector Machine Confusion Matrix')
plt.show()


print("\n",classification_report(svc_prediction, y_test))


"""Step 6: StackingClassifer"""
stack = StackingClassifier(
    estimators=[
        ('lr', DecisionTreeClassifier(random_state=42)),
        ('rf', SVC())
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5
)
stack.fit(X_train, y_train)


stack_prediction = stack.predict(X_test)
print("\nStack Accuracy Score:", accuracy_score(stack_prediction, y_test))


fig, ax=plt.subplots(1,1,figsize=(6, 6), dpi = 500)
ConfusionMatrixDisplay.from_predictions(stack_prediction, 
                                            y_test,cmap=plt.cm.BuPu,ax=ax)
ax.set_title('Stack Confusion Matrix')
plt.show()

"""Step 7: Model Evaluation"""
print("\n",classification_report(stack_prediction, y_test))

joblib.dump(grid_search_dt, "Model.joblib")

decision_tree_loaded=joblib.load("Model.joblib")

test=pd.read_csv("Project_1_Data_Test.csv")


scaled_data_testing = my_scaler.transform(test.iloc[:,0:3])
scaled_data_test_df = pd.DataFrame(scaled_data_testing, 
                                    columns=test.columns[0:3])
X_testing = scaled_data_test_df.join(test.iloc[:,3:])

predictions = decision_tree_loaded.predict(X_testing)
print(predictions)