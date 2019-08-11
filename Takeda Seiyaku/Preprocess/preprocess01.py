import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np

def normarization(df):
    n_df=((df - df.mean()) / df.std(ddof=0))
    return n_df

def preprocess():
    test_df=pd.read_csv("../DataSets/test.csv",sep=",")
    train_df=pd.read_csv("../DataSets/train.csv",sep=",")

    label=train_df[["Score"]]

    train_df.fillna(1e-7)
    test_df.fillna(1e-7)
    #train_df=train_df.dropna(how="all",axis=1)
    #test_df=test_df.dropna(how="all",axis=1)

    train_Id=train_df["ID"]
    test_Id=test_df["ID"]

    norm_label=normarization(train_df[["Score"]])

    del train_df["Score"],test_df["ID"],train_df["ID"]

    n_train=normarization(train_df)
    n_test=normarization(test_df)

    n_train=n_train.fillna(1e-7)
    n_test=n_test.fillna(1e-7)

    np.save("../DataSets/label.npy",label.values)
    np.save("../DataSets/norm_label.npy", norm_label.values)
    np.save("../DataSets/train.npy",n_train.values)
    np.save("../DataSets/test.npy", n_test.values)
class Datasets:
    def __init__(self,label_path,train_path,test_path,):
        self.train=np.load(train_path)
        self.test=np.load(test_path)
        self.label=np.load(label_path)

        self.rand = np.random.permutation(np.arange(np.shape(self.train)[0]))
        self.step=0

    def shuffle(self):
        self.rand=np.random.permutation(np.arange(np.shape(self.train)[0]))



    def batch_data(self,batch_size):
        train=[]
        label=[]
        for i in self.rand[self.step:self.step+batch_size]:
            label.append(self.label[i])
            train.append(self.train[i])
        self.step+=batch_size
        return [train,label]

    def test_data(self):
        return self.test




if __name__=="__main__":
    preprocess()
    df=np.load("../DataSets/train.npy")
    print(np.shape(df))



