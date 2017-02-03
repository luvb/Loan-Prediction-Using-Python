
import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv("C:/Self learning/Projects/Loan Prediction/train.csv") 
test = pd.read_csv("C:/Self learning/Projects/Loan Prediction/test.csv") 

#continuos variables--> Applicant_income , Coapplicant_income , LoanAmount , loanAmount_term
#Categorical variables --> Gender, Married, Dependents, Education, Self_employed,Credit_hist
# property_Area, loan_Status

#printing first 10 observations
df.head(10)

#describing variables for missing values, summary statistic( count, mean,std deviation, min, max,percentiles)
df.describe()
#==========================================================================
#Exploratory data analysis for continuos variables

df['ApplicantIncome'].hist(bins=50) #skewed 

df.boxplot(column='ApplicantIncome') #Outliers present

df['CoapplicantIncome'].hist(bins=50) #skewed 
df.boxplot(column='CoapplicantIncome') #Outliers present

df['LoanAmount'].hist(bins=50) #slightly skewed 
df.boxplot(column='LoanAmount') #Outliers present

df['Loan_Amount_Term'].hist(bins=40) #most of the people take for 360 months loan
df.boxplot(column='Loan_Amount_Term') 
#===========================================================================

#Exploratory data analysis for Categorical variables
df['Gender'].value_counts() # Mostly Male, around (614-601) = 13 values missing

df['Married'].value_counts() # 3 missing values

df['Dependents'].value_counts() 

df['Education'].value_counts() # most of the people are graduates

df['Self_Employed'].value_counts() #Most of the people are not self employed, they are working

df['Credit_History'].value_counts() # most of the people have credit history, missing values

df['Property_Area'].value_counts() 

df['Loan_Status'].value_counts() 

#===========================================================================

temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print( 'Frequency Table for Credit History:' )
print (temp1)

print ('\nProbility of getting loan for each Credit History class:' )
print (temp2)


#========================================================================
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
temp2.plot(kind='bar')

#=======================================================================

#Data/feature engineering
# Missing value treatment

df.apply(lambda x: sum(x.isnull()),axis=0) # lambda is used to create a function on the fly and
#dont have to define any function
 
#impute loan amount column based on self emplyoed and education

#impute self empoloyed first
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No',inplace=True) #impute missing value with No (median)

# imputing missing value of loan amount based on Self_Employed and Education using pivot approach 

table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
print(table)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

df['Married'].value_counts()
df['Married'].fillna('Yes',inplace=True) #impute missing value with No (median)

df['Dependents'].value_counts()
df['Dependents'].fillna('0',inplace=True) #impute missing value with No (median)


df['Gender'].value_counts()
df['Gender'].fillna('Male',inplace=True) #impute missing value with No (median)

df['Credit_History'].value_counts()
df['Credit_History'].fillna(1.0,inplace=True) #impute missing value with No (median)



df['Loan_Amount_Term'].value_counts()
df['Loan_Amount_Term'].fillna(360.0,inplace=True) #impute missing value with No (median)




#==========================================================================================

# checking for distribution of continuous variables and transform them on the same scale

df['LoanAmount'].hist(bins=20)

#log transformation

df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

 # Combine applicant income and coapplicant income and then do log transformation 
 
df['ApplicantIncome'].hist(bins=20)
df['CoapplicantIncome'].hist(bins=20)

df.boxplot(column="ApplicantIncome",by="Loan_Status")
df.hist(column="ApplicantIncome",by="Loan_Status",bins=30)

df.boxplot(column="CoapplicantIncome",by="Loan_Status")
df.hist(column="CoapplicantIncome",by="Loan_Status",bins=30)



df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20) 

df.boxplot(column='LoanAmount_log') #Outliers present

df['Loan_Status'].value_counts()


df.dtypes

#==================================================================
#Building predictive models-->Logistic Regression, Decision Tree, RandomForest, K-Fold cross validation (esimate of test error)

# First we need to encode levels in the categorical variables to numeric using LabelEncoder function

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes 

#====================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, dtest, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
   #Predict on testing data:
  dtest[outcome] = model.predict(dtest[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : " + str(accuracy))
  
  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print ("Cross-Validation Score :" + str((np.mean(error))))
  
  dtest.to_csv("C:/Self learning/Projects/Loan Prediction/submission.csv")

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 
  
#=========================================================================================


#Logistic Regression--->Baseline
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, df,predictor_var,outcome_var)


#Logistic regression-->significant variables
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Gender','Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, df,predictor_var,outcome_var)


#Decision Tree
outcome_var = 'Loan_Status'
model = DecisionTreeClassifier()
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']

classification_model(model, df,predictor_var,outcome_var)
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
featimp.plot(kind='bar', title='Feature Importances')

#Decision Tree--> with high important features based on feature importance graph obtained
outcome_var = 'Loan_Status'
model = DecisionTreeClassifier()
predictor_var = ['Credit_History', 'LoanAmount_log','TotalIncome_log']

classification_model(model, df,predictor_var,outcome_var)
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
featimp.plot(kind='bar', title='Feature Importances')

#Random Forest
outcome_var = 'Loan_Status'
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, df,test,predictor_var,outcome_var)
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
featimp.plot(kind='bar', title='Feature Importances')

#Random Forest--> buit model using high feaure importance so that model generalizes well
#overfiitng is avoided using two approach:
    # 1. Reducing the number of predictors
    # 2. Tuning the model parameters
outcome_var = 'Loan_Status'
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, df,test,predictor_var,outcome_var)

