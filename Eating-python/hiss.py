# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('C:\\Users\\suppo\\Downloads\\Python-project-main\\Python-project-main\\Eating-python\\diabetes_dataset.csv')

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
sns.histplot(data['Glucose'], kde=True, color='blue')
plt.title('Distribution of Glucose Level')
plt.show()

# Scatter plot to explore relationships between variables
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=data)
plt.title('Glucose Level vs BMI')
plt.show()

# Boxplot to explore outliers
sns.boxplot(x='Outcome', y='Glucose', data=data)
plt.title('Boxplot of Glucose Level by Outcome')
plt.show()

# Data Manipulation
# 1. Filter rows where Glucose is greater than 120
filtered_data = data[data['Glucose'] > 120]
print("\nFiltered Data (Glucose > 120):")
print(filtered_data.head())

# 2. Sort data by Glucose level in descending order
sorted_data = data.sort_values(by='Glucose', ascending=False)
print("\nData sorted by Glucose level:")
print(sorted_data.head())

# 3. Create a new column for Glucose levels: 'High' if glucose > 140, else 'Normal'
data['Glucose_Level'] = np.where(data['Glucose'] > 140, 'High', 'Normal')
print("\nData with new Glucose_Level column:")
print(data[['Glucose', 'Glucose_Level']].head())

# 4. Create a new column for BMI category
data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 24.9, 29.9, 40], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
print("\nNew Data with BMI Category:")
print(data.head())

# 5. Grouping data by outcome and calculating mean of Glucose and BMI
grouped_data = data.groupby('Outcome').agg({'Glucose': 'mean', 'BMI': 'mean'})
print("\nGrouped Data by Outcome:")
print(grouped_data)

# 6. Correlation matrix to analyze relationships between numerical features
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
sns.pairplot(data, hue='Outcome')
plt.title('Pairplot of Features by Outcome')
plt.show()

# Report Findings
# You can provide insights based on the analysis here

# Save the manipulated dataset to a CSV file (optional)
# data.to_csv('manipulated_diabetes_dataset.csv', index=False)

