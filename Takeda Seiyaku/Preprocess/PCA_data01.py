import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def normarization(df):
    n_df=((df - df.mean()) / df.std(ddof=0))
    return n_df
def norm_positive(df):
    n_df=df-df.min()
    return n_df

train_path = "../DataSets/train.npy"

pca=PCA()
data=np.load(train_path)
pca.fit(data)

m_com=0
n_com=0
var=pca.explained_variance_
for i in range(len(var)):
    z=np.sum(var[:i])/np.sum(var)
    print(z)
    if z>=0.9:
        n_com=i
        break



print(n_com)
print(np.shape(pca.explained_variance_))

pca_=PCA(n_components=n_com)
pca_.fit(data)
X=pca_.transform(data)
print(np.shape(X))
print(True in np.isnan(X))
X[np.isnan(X)]=0
print(True in np.isnan(X))

df=pd.DataFrame(X)
df_norm=normarization(df)
df=norm_positive(df)      # Mean is positive
data_sets=df_norm.values
np.save("../DataSets/train_c_r09_p.npy",data_sets)



