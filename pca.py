import numpy as np
from pandas import read_csv
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd

orig_df=pd.read_csv('Site674_mice_8inchSMS_nowind_pca.csv', sep=',',header=None)
df=orig_df.drop(orig_df.columns[[0]], axis=1)
colnames = df.values[0]
# Load the breast cancer dataset
#dataset = datasets.load_breast_cancer()
print(colnames)

# Load the features
X = df.values[1:43849]

print((X.shape))
# Create a scaler object
sc = StandardScaler()

# Fit the scaler to the features and transform
X_std = sc.fit_transform(X[0:43848])
print((X_std.shape))
# Create a pca object with the 2 components as a parameter
pca = decomposition.PCA(n_components=8)

# Fit the PCA and transform the data
X_std_pca = pca.fit_transform(X_std)

print(np.round(X_std_pca, 2))
np.savetxt("Site674_mice_pca_8inchSMS_nowind_pca.csv", X_std_pca, delimiter=",")