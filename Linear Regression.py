#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# ### Data Cleaning

# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


train_data.isnull().sum()


# In[6]:


# Extract columns with null values
columns_with_null = train_data.columns[train_data.isnull().any()]

# Display columns with null values
print("Columns with null values:")
for col in columns_with_null:
    print(col)


# In[7]:


# Check for missing values in each column
missing_values = train_data.isnull().sum()

# Print columns with missing values and their corresponding counts
columns_with_missing_values = missing_values[missing_values > 0]
print("\nColumns with Missing Values:")
print(columns_with_missing_values)


# In[8]:


# Check for duplicate rows
duplicates_before = train_data.duplicated().sum()

# Remove duplicate rows
train_data.drop_duplicates(inplace=True)

# Check for duplicate rows after removal
duplicates_after = train_data.duplicated().sum()

# Print the results
if duplicates_before > 0:
    print(f"Handling Duplicates\n{duplicates_before} duplicate row(s) were found and removed.")
else:
    print("Handling Duplicates\nNo duplicate rows found in the dataset.")


# In[9]:


# Get the column names and data types
column_info = train_data.dtypes

# Display column names and data types horizontally
for col_name, data_type in column_info.items():  # Use items() instead of iteritems()
    print(f"{col_name}: {data_type}\t", end='')


# In[10]:


train_data.columns


# In[25]:


# list of categorical columns to exclude
categorical_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# Select numerical columns
numerical_columns = [col for col in train_data.columns if col not in categorical_columns]

# Create a DataFrame with only numerical features and the target variable
numerical_data = train_data[numerical_columns + ['SalePrice']]

# Calculate the correlation matrix
correlation_matrix = numerical_data.corr()

# Step 2: Generate a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix[['SalePrice']], annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation with SalePrice")
plt.show()


# In[26]:


train_data.head()


# In[12]:


# Split the data into training and testing sets
X = train_data[['TotalBsmtSF', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath',"FullBath", "HalfBath"]]
y = train_data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Select the columns of interest (features and target)
features = train_data[['TotalBsmtSF', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']]
target = train_data[['SalePrice']]

# Create a new DataFrame with only the selected columns
data_subset = pd.concat([features, target], axis=1)  # Use square brackets and specify axis=1

# Calculate the correlation matrix
correlation_matrix = data_subset.corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation Heatmap: Features vs. Target")
plt.show()


# In[14]:


# Check for missing values in the training dataset
missing_train = train_data.isnull().sum()
print("Missing Values in Training Data:")
print(missing_train)

# Check for missing values in the testing dataset
missing_test = test_data.isnull().sum()
print("\nMissing Values in Testing Data:")
print(missing_test)


# In[15]:


# Missing values in selected features
features_missing_values = X.isnull().sum()
features_missing_values


# In[31]:


#create a linear regression model
model = LinearRegression()


# In[32]:


#Fit the model to the training data
model.fit(X_train, y_train)
print(model)


# In[33]:


# Make predictions on the test data
y_pred = model.predict(X_test)


# In[34]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[35]:


print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[36]:


X.sample(6)


# In[37]:


# Predict the price of a new house
new_house = np.array([[1564, 4, 2,0,1,2]]) 
predicted_price = model.predict(new_house)
print(f"Predicted Price for the New House: ${predicted_price[0]:.2f}")


# In[38]:


# Cross-validation to assess model performance
cv_scores = cross_val_score(model, X, y, cv=5)
print('Cross-Validation Scores:', cv_scores)
print('Mean CV Score:', cv_scores.mean())


# In[39]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()


# In[ ]:





# In[ ]:




