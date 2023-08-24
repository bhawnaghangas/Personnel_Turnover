#!/usr/bin/env python
# coding: utf-8

# Personnel Turnover Project by BHAWNA GHANGAS

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import seaborn as sns


# In[2]:


dummy = pd.read_csv(r"C:\Users\ghang\Downloads\HR_Employee_Attrition_Data.csv", na_values =['NA'])
temp = dummy.columns.values
temp


# In[3]:


dummy.head()


# In[5]:


new_df= dummy


# In[6]:


new_df.apply(lambda x: x.isnull().sum())


# In[7]:


def preprocessor(df):
    res_df = df.copy()
    le = preprocessing.LabelEncoder()
    #StandardHours JobLevel YearsInCurrentRole YearsWithCurrManager
    #MonthlyIncome over18 employee number    
    res_df['Age'] = le.fit_transform(res_df['Age'])
    res_df['Attrition'] = le.fit_transform(res_df['Attrition'])
    res_df['BusinessTravel'] = le.fit_transform(res_df['BusinessTravel'])
    res_df['DailyRate'] = le.fit_transform(res_df['DailyRate'])
    res_df['Department'] = le.fit_transform(res_df['Department'])
    res_df['DistanceFromHome'] = le.fit_transform(res_df['DistanceFromHome'])
    res_df['Education'] = le.fit_transform(res_df['Education'])
    res_df['EducationField'] = le.fit_transform(res_df['EducationField'])
    res_df['EmployeeCount'] = le.fit_transform(res_df['EmployeeCount'])
    res_df['EmployeeNumber'] = le.fit_transform(res_df['EmployeeNumber'])
    res_df['EnvironmentSatisfaction'] = le.fit_transform(res_df['EnvironmentSatisfaction'])
    res_df['Gender'] = le.fit_transform(res_df['Gender'])
    res_df['HourlyRate'] = le.fit_transform(res_df['HourlyRate'])
    res_df['JobInvolvement'] = le.fit_transform(res_df['JobInvolvement'])
    res_df['JobLevel'] = le.fit_transform(res_df['JobLevel'])
    res_df['JobRole'] = le.fit_transform(res_df['JobRole'])
    res_df['JobSatisfaction'] = le.fit_transform(res_df['JobSatisfaction'])
    res_df['MaritalStatus'] = le.fit_transform(res_df['MaritalStatus'])
    res_df['MonthlyIncome'] = le.fit_transform(res_df['MonthlyIncome'])
    res_df['MonthlyRate'] = le.fit_transform(res_df['MonthlyRate'])
    res_df['NumCompaniesWorked'] = le.fit_transform(res_df['NumCompaniesWorked'])
    res_df['Over18'] = le.fit_transform(res_df['Over18'])
    res_df['OverTime'] = le.fit_transform(res_df['OverTime'])
    res_df['PercentSalaryHike'] = le.fit_transform(res_df['PercentSalaryHike'])
    res_df['PerformanceRating'] = le.fit_transform(res_df['PerformanceRating'])
    res_df['RelationshipSatisfaction'] = le.fit_transform(res_df['RelationshipSatisfaction'])
    res_df['StandardHours'] = le.fit_transform(res_df['StandardHours'])
    res_df['StockOptionLevel'] = le.fit_transform(res_df['StockOptionLevel'])
    res_df['TotalWorkingYears'] = le.fit_transform(res_df['TotalWorkingYears'])
    res_df['TrainingTimesLastYear'] = le.fit_transform(res_df['TrainingTimesLastYear'])
    res_df['WorkLifeBalance'] = le.fit_transform(res_df['WorkLifeBalance'])
    res_df['YearsAtCompany'] = le.fit_transform(res_df['YearsAtCompany'])
    res_df['YearsInCurrentRole'] = le.fit_transform(res_df['YearsInCurrentRole'])
    res_df['YearsSinceLastPromotion'] = le.fit_transform(res_df['YearsSinceLastPromotion'])
    res_df['YearsWithCurrManager'] = le.fit_transform(res_df['YearsWithCurrManager'])
    return res_df


# In[13]:


en_df = preprocessor(new_df)


# In[14]:


en_df.drop('PercentSalaryHike', inplace=True, axis=1)


# In[15]:


en_df.drop('EmployeeCount', inplace=True, axis=1)


# In[16]:


en_df.drop('StandardHours', inplace=True,axis=1)                  


# In[17]:


en_df.drop('JobLevel', inplace=True,axis=1)


# In[18]:


en_df.drop('YearsInCurrentRole', inplace=True,axis=1)


# In[19]:


en_df.drop('YearsWithCurrManager', inplace=True,axis=1)


# In[20]:


#en_df.drop('MonthlyIncome', inplace=True,axis=1)


# In[23]:


en_df.drop('OverTime', inplace=True,axis=1)


# In[24]:


en_df.drop('Gender', inplace=True,axis=1)


# In[ ]:





# In[ ]:





# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')

import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from urllib.request import urlopen 

plt.style.use('ggplot')
pd.set_option('display.max_columns', 500) 


# In[26]:


print(en_df['Attrition'].unique())


# In[27]:


feature_space = en_df.iloc[:, en_df.columns != 'Attrition']
feature_class = en_df.iloc[:, en_df.columns == 'Attrition']


# In[28]:


training_set, test_set, class_set, test_class_set = train_test_split(feature_space,
                                                                    feature_class,
                                                                    test_size = 0.20, 
                                                                    random_state = 42)


# In[29]:


# Cleaning test sets to avoid future warning messages
class_set = class_set.values.ravel() 
test_class_set = test_class_set.values.ravel() 


# In[30]:


# Set the random state for reproducibility
fit_rf = RandomForestClassifier(random_state=42)


# In[23]:


np.random.seed(42)
start = time.time()

param_dist = {'max_depth': [2,3,4],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']
              }

cv_rf = GridSearchCV(fit_rf, cv = 10,
                     param_grid=param_dist, 
                     n_jobs = 2)

cv_rf.fit(training_set, class_set)
print('Best Parameters using grid search: \n', cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))


# In[31]:


# Set best parameters given by grid search 
fit_rf.set_params(criterion = 'gini',
                  max_features = 'log2', 
                  max_depth = 4)


# In[32]:


fit_rf.set_params(warm_start=True, 
                  oob_score=True)

min_estimators = 15
max_estimators = 2000

error_rate = {}

for i in range(min_estimators, max_estimators + 1):
    fit_rf.set_params(n_estimators=i)
    fit_rf.fit(training_set, class_set)

    oob_error = 1 - fit_rf.oob_score_
    error_rate[i] = oob_error


# In[33]:


# Convert dictionary to a pandas series for easy plotting 
oob_series = pd.Series(error_rate)


# In[34]:


fig, ax = plt.subplots(figsize=(10,2))

ax.set_facecolor('#fafafa')

oob_series.plot(kind='line',color = 'red')
plt.axhline(0.055, color='#875FDB',linestyle='--')
plt.axhline(0.05, color='#875FDB',linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')


# In[37]:


print('OOB Error rate for 500 trees is: {0:.5f}'.format(oob_series[800]))


# In[38]:


# Refine the tree via OOB Output
fit_rf.set_params(n_estimators=800,
                  bootstrap = True,
                  warm_start=False, 
                  oob_score=False)


# In[39]:


fit_rf.fit(training_set, class_set)


# In[40]:


def variable_importance(fit):
    """
    Purpose
    ----------
    Checks if model is fitted CART model then produces variable importance
    and respective indices in dictionary.

    Parameters
    ----------
    * fit:  Fitted model containing the attribute feature_importances_

    Returns
    ----------
    Dictionary containing arrays with importance score and index of columns
    ordered in descending order of importance.
    """
    try:
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit)) 

        # Captures whether the model has been trained
        if not vars(fit)["estimators_"]:
            return print("Model does not appear to be trained.")
    except KeyError:
        print("Model entered does not contain 'estimators_' attribute.")

    importances = fit.feature_importances_
    indices = np.argsort(importances)[::-1]
    return {'importance': importances,
            'index': indices}


# In[41]:


var_imp_rf = variable_importance(fit_rf)

importances_rf = var_imp_rf['importance']

indices_rf = var_imp_rf['index']


# In[42]:


names = ['Age', 'Attrition', 'BusinessTravel', 
         'DailyRate', 'Department', 'DistanceFromHome', 
         'Education', 'EducationField', 
         'EmployeeCount','EmployeeNumber', 
         'EnvironmentSatisfaction', 'Gender',
         'HourlyRate', 'JobInvolvement', 'JobLevel', 
         'JobRole', 'JobSatisfaction', 'MaritalStatus', 
         'MonthlyIncome', 'MonthlyRate', 
         'NumCompaniesWorked', 'Over18', 
         'OverTime', 'PercentSalaryHike', 
         'PerformanceRating', 'RelationshipSatisfaction', 
         'StandardHours', 'StockOptionLevel', 
         'TotalWorkingYears', 'TrainingTimesLastYear', 
         'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager'] 

dx = ['Benign', 'Malignant']


# In[43]:


names_index = names[2:]


# In[44]:


def print_var_importance(importance, indices, name_index):
    """
    Purpose
    ----------
    Prints dependent variable names ordered from largest to smallest
    based on information gain for CART model.
    Parameters
    ----------
    * importance: Array returned from feature_importances_ for CART
                models organized by dataframe index
    * indices: Organized index of dataframe from largest to smallest
                based on feature_importances_
    * name_index: Name of columns included in model

    Returns
    ----------
    Prints feature importance in descending order
    """
    print("Feature ranking:")

    for f in range(0, indices.shape[0]):
        i = f
        print("{0}. The feature '{1}' has a Mean Decrease in Impurity of {2:.5f}"
              .format(f + 1,
                      names_index[indices[i]],
                      importance[indices[f]]))


# In[45]:


print_var_importance(importances_rf, indices_rf, names_index)


# In[46]:


en_df.set_index(['EmployeeNumber'], inplace = True) 


# In[ ]:





# In[47]:


def variable_importance_plot(importance, indices, name_index):
    """
    Purpose
    ----------
    Prints bar chart detailing variable importance for CART model
    NOTE: feature_space list was created because the bar chart
    was transposed and index would be in incorrect order.

    Parameters
    ----------
    * importance: Array returned from feature_importances_ for CART
                models organized by dataframe index
    * indices: Organized index of dataframe from largest to smallest
                based on feature_importances_
    * name_index: Name of columns included in model

    Returns:
    ----------
    Returns variable importance plot in descending order
    """
    index = np.arange(len(names_index))

    importance_desc = sorted(importance)
    feature_space = []
    for i in range(indices.shape[0] - 1, -1, -1):
        feature_space.append(names_index[indices[i]])

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_axis_bgcolor('#fafafa')
    plt.title('Feature importances for Random Forest Model    \nBreast Cancer (Diagnostic)')
    plt.barh(index,
             importance_desc,
             align="center",
             color = '#875FDB')
    plt.yticks(index,
               feature_space)

    plt.ylim(-1, 30)
    plt.xlim(0, max(importance_desc) + 0.01)
    plt.xlabel('Mean Decrease in Impurity')
    plt.ylabel('Feature')

    plt.show()
    plt.close()


# In[48]:


variable_importance_plot(importances_rf, indices_rf, names_index)


# In[49]:


predictions_rf = fit_rf.predict(test_set)


# In[50]:


test_crosst=0
def create_conf_mat(test_class_set, predictions):
    """Function returns confusion matrix comparing two arrays"""
    if (len(test_class_set.shape) != len(predictions.shape) == 1):
        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
    elif (test_class_set.shape != predictions.shape):
        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')
    else:
        # Set Metrics
        test_crosstb_comp = pd.crosstab(index = test_class_set,
                                        columns = predictions)
        # Changed for Future deprecation of as_matrix
        test_crosstb = test_crosstb_comp.values
        return test_crosst


# In[ ]:





# In[52]:


accuracy_rf = fit_rf.score(test_set, test_class_set)

print("Here is our mean accuracy on the test set:\n {0:.3f}"      .format(accuracy_rf))


# In[53]:


# Here we calculate the test error rate!
test_error_rate_rf = 1 - accuracy_rf
print("The test error rate for our model is:\n {0: .4f}"      .format(test_error_rate_rf))


# In[54]:


# We grab the second array from the output which corresponds to
# to the predicted probabilites of positive classes 
# Ordered wrt fit.classes_ in our case [0, 1] where 1 is our positive class
predictions_prob = fit_rf.predict_proba(test_set)[:, 1]

fpr2, tpr2, _ = roc_curve(test_class_set,
                          predictions_prob,
                          pos_label = 1)


# In[55]:


auc_rf = auc(fpr2, tpr2)


# In[56]:


def plot_roc_curve(fpr, tpr, auc, estimator, xlim=None, ylim=None):
    """
    Purpose
    ----------
    Function creates ROC Curve for respective model given selected parameters.
    Optional x and y limits to zoom into graph

    Parameters
    ----------
    * fpr: Array returned from sklearn.metrics.roc_curve for increasing
            false positive rates
    * tpr: Array returned from sklearn.metrics.roc_curve for increasing
            true positive rates
    * auc: Float returned from sklearn.metrics.auc (Area under Curve)
    * estimator: String represenation of appropriate model, can only contain the
    following: ['knn', 'rf', 'nn']
    * xlim: Set upper and lower x-limits
    * ylim: Set upper and lower y-limits
    """
    my_estimators = {'knn': ['Kth Nearest Neighbor', 'deeppink'],
              'rf': ['Random Forest', 'red'],
              'nn': ['Neural Network', 'purple']}

    try:
        plot_title = my_estimators[estimator][0]
        color_value = my_estimators[estimator][1]
    except KeyError as e:
        print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
        raise

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#fafafa')

    plt.plot(fpr, tpr,
             color=color_value,
             linewidth=1)
    plt.title('ROC Curve For {0} (AUC = {1: 0.3f})'              .format(plot_title, auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
    plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
    plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.close()


# In[57]:


plot_roc_curve(fpr2, tpr2, auc_rf, 'rf',
               xlim=(-0.01, 1.05), 
               ylim=(0.001, 1.05))


# In[58]:


#dx = ['Benign', 'Malignant']
dx=['Yes','No']


# In[59]:



def print_class_report(predictions, alg_name):
    """
    Purpose
    ----------
    Function helps automate the report generated by the
    sklearn package. Useful for multiple model comparison

    Parameters:
    ----------
    predictions: The predictions made by the algorithm used
    alg_name: String containing the name of the algorithm used
    
    Returns:
    ----------
    Returns classification report generated from sklearn. 
    """
    print('Classification Report for {0}:'.format(alg_name))
    print(classification_report(predictions, 
            test_class_set, 
            target_names = dx))


# In[60]:


class_report = print_class_report(predictions_rf, 'Random Forest')


# Personnel Turnover Project by BHAWNA GHANGAS
