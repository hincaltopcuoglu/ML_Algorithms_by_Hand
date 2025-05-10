import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample data
X  = np.array([[1,2],[3,4],[5,6],[7,8]])
y = np.array([0,0,1,1])

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=54)

#Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

### Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# logistic regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled,y_train)
log_reg.predict(X_test_scaled)


# Decision tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train_scaled,y_train)
tree.predict(X_test_scaled)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)

#####################################################################

### Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
lin_reg_pred = lin_reg.predict(X_test)

# Random forest regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train,y_train)
rf_reg_pred = rf_reg.predict(X_test)

#####################################################################

### Model Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

# Classifcation Metrics
accuracy = accuracy_score(y_test,rf_pred)
report = classification_report(y_test,rf_pred)
conf_matrix = confusion_matrix(y_test, rf_pred)

# Regression Metrics
mse = mean_squared_error(y_test,lin_reg_pred)
r2 = r2_score(y_test, lin_reg_pred)



#####################################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


'''
Problem: Predicting Diabetes
We'll use the Pima Indians Diabetes dataset to predict whether a patient has diabetes based on diagnostic measurements.
'''

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(url, names=column_names)

#display the first few rows
print(df.head())

# check missing values
print("\Missing Values:")
print(df.isnull().sum())


# basic statistics
print("\Basic Statistics")
print(df.describe())


# prepapre features and target
X  = df.drop('Outcome',axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# train logistic regression model

log_reg = LogisticRegression(random_state=42, max_iter=200)
log_reg.fit(X_train_scaled,y_train)
log_reg_pred = log_reg.predict(X_test_scaled)

# Evaluate Model
print("\nLogistic Regression Results:")
print(f" Accuracy: {accuracy_score(y_test,log_reg_pred)}")

print("\n Classification Report:")
print(classification_report(y_test,log_reg_pred,))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,log_reg_pred))


# Cross-validation
cv_scores = cross_val_score(log_reg, X_train_scaled,y_train, cv=5)
print(f"\nCross-validation scores {cv_scores}")
print(f"Mean CV Score:{cv_scores.mean():.3f}")

# get feature importance for logistic regression coefficients

# get coefficients
coefficients = log_reg.coef_[0]

# create a dataframe of features and their coeffiients
feature_importance = pd.DataFrame({
    'Feature:': X.columns,
    'Coefficient': coefficients,
    'Absolute Coefficients': np.abs(coefficients)
})

feature_importance = feature_importance.sort_values(by='Absolute Coefficients',ascending=False)

print("Feature Importance based on Logistic Regression Coefficients:")
print(feature_importance)