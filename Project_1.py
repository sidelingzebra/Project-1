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



sb.set_theme(style="darkgrid")

"""Step 1: Data In"""
df=pd.read_csv("Project_1_Data.csv")

#Converting Step column from Integar to 
#df['Step']=df['Step'].astype(float)

step_cat = df[["Step"]]
step_cat.head(4)

ordinal_encoder = OrdinalEncoder()
step_cat_encoded = ordinal_encoder.fit_transform(step_cat)
step_cat_encoded[:13]

step_cat_encoder = OneHotEncoder()
step_1hot = step_cat_encoder.fit_transform(step_cat)

step_1hot.toarray()



"""Step 2: Visalizing Data"""
    #Before visalizing, split data between train & test
    
    #Creates a new column that splits the data for Stratified Random Sampling
Split_1=StratifiedShuffleSplit (n_splits=1,test_size=0.2,random_state=42)
df["Split"] = pd.cut(df["Step"],
                          bins=[0, 2, 4, 6, np.inf],
                          labels=[1, 2, 3, 4])

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
fig4.set_title('Y Regression')
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
scaled_data_train_df = pd.DataFrame(scaled_data_train, 
                                    columns=X_train.columns[0:3])
X_train = scaled_data_train_df.join(X_train.iloc[:,3:])

scaled_data_test = my_scaler.transform(X_test.iloc[:,0:3])
scaled_data_test_df = pd.DataFrame(scaled_data_test, 
                                   columns=X_test.columns[0:3])
X_test = scaled_data_test_df.join(X_test.iloc[:,3:])


#Print the Correlation Matrix
plt.figure(figsize=(6, 6),dpi = 500)
corr_matrix = (X_train).corr()
fig6=sb.heatmap(np.abs(corr_matrix),annot=True)
fig6.set_title('Correlation matrix')
#sb.pairplot(strat_df_train)




"""ML Model 1 - Linear Regression"""
linear_reg = LogisticRegression()
param_grid_lr = {}  # No hyperparameters to tune for plain linear regression, but you still apply GridSearchCV.
grid_search_lr = GridSearchCV(linear_reg, param_grid_lr, cv=5, 
                              scoring='neg_mean_absolute_error', 
                              n_jobs=-1)

grid_search_lr.fit(X_train, y_train)

best_model_lr = grid_search_lr.best_estimator_
print("\nBest Linear Regression Model:", best_model_lr)

lr_prediction = grid_search_lr.predict(X_test)
print("\nLogistic Regression Accuracy Score:", accuracy_score(lr_prediction, y_test))

print(confusion_matrix(lr_prediction, y_test))
print(classification_report(lr_prediction, y_test))





"""ML Model 2 - Random Forest"""

forest_reg = RandomForestClassifier()

param_grid_rf = [
  {'n_estimators': [3, 10, 30],
   'max_features': [2, 4, 6, 8]},
  {'bootstrap': [False], 
   'n_estimators': [3, 10], 
   'max_features': [2, 3, 4]},
  ]

grid_search_rf = GridSearchCV(forest_reg, param_grid_rf, cv=5,
                            scoring='neg_mean_squared_error',
                            return_train_score=True)

grid_search_rf.fit(X_train,y_train)

best_model_rf = grid_search_rf.best_estimator_
print("\nBest Random Forest Model:", best_model_rf)

feature_importances_rf = grid_search_rf.best_estimator_.feature_importances_
print("\nRandom Forest Feature Importance:", feature_importances_rf)

rf_prediction = grid_search_rf.predict(X_test)
print("\nRandom Forest Accuracy Score:", accuracy_score(rf_prediction, y_test))

print(confusion_matrix(rf_prediction, y_test))
print(classification_report(rf_prediction, y_test))





"""ML Model 3 - Decision Tree"""
decision_tree = DecisionTreeClassifier(random_state=24)

param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = RandomizedSearchCV(decision_tree, param_grid_dt, cv=5,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1)

grid_search_dt.fit(X_train,y_train)

best_model_dt = grid_search_dt.best_estimator_
print("\nBest Decision Tree Model:", best_model_dt)

feature_importances_dt = grid_search_dt.best_estimator_.feature_importances_
print("\nDecision Tree Feature Importance:", feature_importances_dt)

dt_prediction = grid_search_dt.predict(X_test)
print("\nDecision Tree Accuracy Score:", accuracy_score(dt_prediction, y_test))

print(confusion_matrix(dt_prediction, y_test))
print(classification_report(dt_prediction, y_test))







"""ML Model 4 - Support Vector Machine"""
svr = SVR()
param_grid_svr = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}

grid_search_svr = GridSearchCV(svr, param_grid_svr, cv=5, 
                               scoring='neg_mean_absolute_error', 
                               n_jobs=-1)

grid_search_svr.fit(X_train, y_train)

best_model_svr = grid_search_svr.best_estimator_
print("\nBest SVM Model:", best_model_svr)



# Training and testing error for Linear Regression
y_train_pred_lr = best_model_lr.predict(X_train)
y_test_pred_lr = best_model_lr.predict(X_test)
mae_train_lr = mean_absolute_error(y_train, y_train_pred_lr)
mae_test_lr = mean_absolute_error(y_test, y_test_pred_lr)
print(f"\nLinear Regression - MAE (Train): {mae_train_lr}, MAE (Test): {mae_test_lr}")

# Training and testing error for Decision Tree
y_train_pred_dt = best_model_dt.predict(X_train)
y_test_pred_dt = best_model_dt.predict(X_test)
mae_train_dt = mean_absolute_error(y_train, y_train_pred_dt)
mae_test_dt = mean_absolute_error(y_test, y_test_pred_dt)
print(f"\nDecision Tree - MAE (Train): {mae_train_dt}, MAE (Test): {mae_test_dt}")

# Training and testing error for Random Forest
y_train_pred_rf = best_model_rf.predict(X_train)
y_test_pred_rf = best_model_rf.predict(X_test)
mae_train_rf = mean_absolute_error(y_train, y_train_pred_rf)
mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)
print(f"\nRandom Forest - MAE (Train): {mae_train_rf}, MAE (Test): {mae_test_rf}")

# Training and testing error for SVM
y_train_pred_svr = best_model_svr.predict(X_train)
y_test_pred_svr = best_model_svr.predict(X_test)
mae_train_svr = mean_absolute_error(y_train, y_train_pred_svr)
mae_test_svr = mean_absolute_error(y_test, y_test_pred_svr)
print(f"\nSVM - MAE (Train): {mae_train_svr}, MAE (Test): {mae_test_svr}")


# #Confusion Matrix through all Step Values
# i=1
# while i<14:
#     y_train_i = (y_train == i) # True for all 5s, False for all other digits
#     y_test_i = (y_test == i)
    
#     sgd_clf = SGDClassifier(random_state=42)
#     sgd_clf.fit(X_train, y_train_i)
    
#     y_train_pred_i = cross_val_predict(sgd_clf, X_train, y_train_i, cv=3)
#     C_mat_i=confusion_matrix(y_train_i, y_train_pred_i)
    
#     f1_score_i=f1_score(y_train_i, y_train_pred_i)

#     print("\nConfusion Matrix for: ", i, "is:\n", C_mat_i)
#     print("\nF1 Score for: ", i, "is:\n", f1_score_i)
#     i=i+1

svc_model = SVC(random_state=24)
svc_model.fit(X_train, y_train) 

svc_prediction = svc_model.predict(X_test)
print("\nSVC Accuracy Score:", accuracy_score(svc_prediction, y_test))
print(confusion_matrix(svc_prediction, y_test))
print(classification_report(svc_prediction, y_test))
