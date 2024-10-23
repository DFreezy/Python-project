# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('C:/Users/Du-Wayne.Frieslaar/Desktop/Something different/diabetes_dataset.csv')




# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Data Cleaning: Handle missing data if there are any
# Uncomment the following line if needed:
# data.fillna(data.mean(), inplace=True)

# Check for duplicates and remove if necessary
print("\nDuplicate Rows: ", data.duplicated().sum())
data.drop_duplicates(inplace=True)

# Data Exploration
# Descriptive statistics for each feature
print("\nDescriptive Statistics:")
print(data.describe())

# Visualize the distribution of a numerical feature (e.g., glucose level)
sns.histplot(data['Glucose'], kde=True, color='blue')  # Replace 'Glucose' with the actual column name
plt.title('Distribution of Glucose Level')
plt.show()

# Scatter plot to explore relationships between variables
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=data)  # Replace 'BMI' and 'Outcome' as needed
plt.title('Glucose Level vs BMI')
plt.show()

# Boxplot to explore outliers
sns.boxplot(x='Outcome', y='Glucose', data=data)  # Replace 'Glucose' as needed
plt.title('Boxplot of Glucose Level by Outcome')
plt.show()

# Data Manipulation
# Create a new column, for example, a BMI category
data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 24.9, 29.9, 40], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
print("\nNew Data with BMI Category:")
print(data.head())

# Grouping data by outcome and calculating mean of each feature
grouped_data = data.groupby('Outcome').mean()
print("\nMean values by Outcome:")
print(grouped_data)

# Correlation matrix to analyze relationships between numerical features
correlation_matrix = data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Data Visualization
# Pairplot to visualize relationships between features and outcomes
sns.pairplot(data, hue='Outcome')  # Replace 'Outcome' as needed
plt.title('Pairplot of Features by Outcome')
plt.show()

# Report Findings
# Summary: Based on the analysis, you can provide insights on relationships, distributions, etc.

# Save your analysis results to a CSV file (optional)
# data.to_csv('analysis_results.csv', index=False)
