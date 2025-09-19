import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read in dataset as a dataframe
df = pd.read_csv('winequality-red.csv')

# Show the first few rows of the dataframe
print(df.head())
# Show the first few columns of the dataframe
print(df.columns)

# Distribution of the 'quality' variable
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=df)
plt.title('Distribution of Wine Quality Ratings')
plt.xlabel('Quality Rating')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Wine Features')
plt.show()

# Check for missing values
print(df.isnull().sum())

# Normalize/Standardize the features (excluding the target variable 'quality')
features = df.columns[:-1]  # Exclude the target variable 'quality'
x = df.loc[:, features].values
y = df.loc[:, ['quality']].values
x = StandardScaler().fit_transform(x)  # Standardizing the features


# Applying PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization purposes
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# Combine PCA result with the target variable
finalDf = pd.concat([principalDf, df[['quality']]], axis=1)

# Show the result of PCA and the variance explained by the 2 principal components
explained_variance = pca.explained_variance_ratio_
print(finalDf.head())
print(explained_variance)