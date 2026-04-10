
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Titanic-Dataset.csv")


print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())


plt.figure()
df['Age'].hist(bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


plt.figure()
sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.show()


plt.figure()
sns.countplot(x='Survived', data=df)
plt.title("Survival Count (0 = No, 1 = Yes)")
plt.show()


plt.figure()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()


plt.figure()
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


sns.pairplot(df)
plt.show()


print("\nBasic Insights:")
print("- Check graphs for patterns like survival rate, age distribution, etc.")