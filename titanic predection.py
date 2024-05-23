import pandas as pd

# Load the Titanic dataset
train = pd.read_csv("C:/Users/shari/Downloads/titanic prediction.csv")

# View the first few rows of the dataset
print(train.head())

# Check for null values
print(train.isnull().sum())

# Visualize survival by gender
import seaborn as sns
import matplotlib.pyplot as plt
sns.catplot(x="Sex", hue="Survived", kind="count", data=train)

# Visualize survival by passenger class
pclass_survived = train.groupby(['Pclass', 'Survived']).size().unstack()
sns.heatmap(pclass_survived, annot=True, fmt="d")

# Explore age distribution
sns.histplot(data=train, x="Age", hue="Survived", kde=True)

plt.show()





