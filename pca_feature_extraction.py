#This script helps to extract features
import numpy as np
from pandas import read_csv
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd

#include appropriate dataset csv file
orig_df=pd.read_csv('Site674_mice.csv', sep=',',header=None)
df=orig_df.drop(orig_df.columns[[0]], axis=1)
colnames = df.values[0]
# Load the breast cancer dataset
#dataset = datasets.load_breast_cancer()
print(colnames)

# Load the features
X = df.values[1:len(df.values)]

# Create a scaler object
sc = StandardScaler()

# Fit the scaler to the features and transform
X_std = sc.fit_transform(X[0:len(df.values)])
# Create a pca object with the 8 components as a parameter
pca = decomposition.PCA(n_components=8)

# Fit the PCA and transform the data
X_std_pca = pca.fit_transform(X_std)


# View the new feature data's shape
dataset = read_csv('Site674_mice.csv', header=0, index_col=0)
values = dataset.values
values = values.astype('float32')
col1 = values[:,0]

#save features to csv file
np.savetxt("Site674_pca.csv", X_std_pca, delimiter=",")

#manually write the PREC.I.1..in. column header and PC1 to PC8 for remaining features
#once done manually copy the 1st column in resulting csv for numbering of rows
