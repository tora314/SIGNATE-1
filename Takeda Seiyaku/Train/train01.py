import tensorflow as tf
import numpy as np
from NeuralNetwork import NN
from tqdm import tqdm
import matplotlib.pyplot as plt
from  datetime import date

class RealtimePlot:
    def __init__(self,x,y):
        self.fig= plt.figure(figsize=(8,8))
        self.ax=self.fig.add_subplot(211)
        self.lines, = self.ax.plot(x, y)


        self.ax2 = self.fig.add_subplot(212)
        self.lines2, = self.ax2.plot(x, y)



    def plot_fig(self,y,y2):
        #plt.cla()
        #self.fig, self.ax = plt.subplots(1, 1)
        x=np.arange(0,len(y))
        self.lines, = self.ax.plot(x, y)

        self.lines.set_data(x, y)


        plt.yscale('log')
        plt.grid(which='major', color='black', linestyle='-')
        plt.grid(which='minor', color='black', linestyle='-')
        plt.xlabel("train num")
        plt.ylabel("loss")

        boxdic = {
            "facecolor": "lightgreen",
            "edgecolor": "darkred",
            "boxstyle": "Round",
            "linewidth": 2
        }


        self.ax.text(0, 1, "test loss = "+str(y[-1]), size=10, transform=self.ax.transAxes,bbox=boxdic)
        #m=np.min(y)

        self.ax.set_xlim([np.min(x), np.max(x)])
        self.ax.set_ylim([-10,1])
        self.ax.set_title("R2")
        plt.cla()


        #-------------------------------------------------


        plt.cla()
        x2=np.arange(0,len(y2))
        self.lines2, = self.ax2.plot(x2, y2)

        self.lines2.set_data(x2, y2)

        plt.yscale('log')
        plt.grid(which='major', color='black', linestyle='-')
        plt.grid(which='minor', color='black', linestyle='-')
        plt.xlabel("train num")
        plt.ylabel("loss")

        boxdic = {
            "facecolor": "lightgreen",
            "edgecolor": "darkred",
            "boxstyle": "Round",
            "linewidth": 2
        }

        self.ax2.text(0, 1, "test loss = " + str(y2[-1]), size=10, transform=self.ax2.transAxes, bbox=boxdic)
        # m=np.min(y)

        self.ax2.set_xlim([np.min(x2), np.max(x2)])
        self.ax2.set_ylim([np.min(y2), np.max(y2)])
        self.ax2.set_title("SSE")



    def show_fig(self,pause=0.0001):
        plt.pause(pause)


class Datasets:
    def __init__(self,label_path,train_path,train_ratio=0.8):
        self.data=np.load(train_path)
        self.data[np.isnan(self.data)]=0
        #self.test=np.load(test_path)
        self.label=np.load(label_path)
        self.label[np.isnan(self.label)]=0


        self.train_ratio=train_ratio

        rand=np.random.randint(0,np.shape(self.data)[0])
        tr_n=round(np.shape(self.data)[0]*train_ratio)


        self.train=self.data[ : ]


        self.step=0

        self.shuffle()

    def shuffle(self):


        rand = np.random.randint(0, np.shape(self.data)[0])
        tr_n = round(np.shape(self.data)[0] * self.train_ratio)


        self.rand = np.random.permutation(np.arange(np.shape(self.train)[0]))


        if rand+tr_n>=np.shape(self.data)[0]:

            self.train = np.append(self.data[rand : ],self.data[: tr_n+rand-np.shape(self.data)[0]],axis=0)
            self.train_label=np.append(self.label[rand : ],self.label[: tr_n+rand-np.shape(self.data)[0]],axis=0)

            self.test = self.data[tr_n + rand - np.shape(self.data)[0] + 1: rand - 1]
            self.test_label = self.label[tr_n + rand - np.shape(self.data)[0] + 1: rand - 1]

        else:
            #print("hello")
            self.train=self.data[rand : tr_n+rand]
            self.train_label = self.label[rand: tr_n + rand]


            self.test = np.append(self.data[tr_n + 1 + rand:], self.data[: rand - 1], axis=0)
            self.test_label=np.append(self.label[tr_n+1+rand:],self.label[: rand-1],axis=0)


        self.step=0


    def batch_data(self,batch_size):
        train=[]
        label=np.asarray([])
        #print(np.shape(self.rand))
        m=0

        for i in self.rand[self.step:self.step+batch_size]:
            #print(i)
            label=np.append(label,self.train_label[i],axis=0)
            train.append(self.train[i])#np.append(train,self.train[i],axis=-1)
            m+=1


        train=np.asarray(train)
        #label=np.reshape(label,[-1])
        self.step+=batch_size
        #print(train)
        self.step+=1
        return [train,label]

    def test_data(self):
        self.test_label=np.reshape(self.test_label,[-1])
        return [self.test,self.test_label]

    def get_len(self):
        return np.shape(self.train)[0]

    def get_size(self):
        #print(np.shape(self.train))
        return np.shape(self.train)[1]



class Model(NN):
    # Template model class
    def __init__(self):
        super().__init__()
        self.sess=tf.InteractiveSession()

        self.histry=[]




    def create_model(self,hidden,output_size,data_size):
        """
        :param hidden: type list 隠れ層のパーセプトロンの数
        :param output_size: type int 出力の次元
        :param data_size: type int データのサイズ
        :return: None
        :self.opt=optimizerを格納するメンバ変数
        :self.loss=lossを格納するメンバ変数
        :self.out=modelの出力を格納するメンバ変数
        :self.label=正解データを格納するメンバ変数
        :self.x=入力を格納するメンバ変数
        :todo : モデルを組み立てる
        """

        x=self.placeholder("x")
        label=self.placeholder("label")
        label=tf.reshape(label,[-1])

        x_ = tf.reshape(x, [-1, data_size])

        n1 = self.elu(x_, self.weight([data_size, hidden[0]]), self.bias([hidden[0]]),drop=0.5)
        n2=self.elu(n1, self.weight([hidden[0], hidden[1]]), self.bias([hidden[1]]),drop=0.5)
        n5 = tf.abs(n2)
        n3=self.elu(n2, self.weight([hidden[1], hidden[2]]), self.bias([hidden[2]]),drop=0.5)
        n4=self.elu(n3, self.weight([hidden[2], hidden[3]]), self.bias([hidden[3]]),drop=0.5)


        out = self.elu(n2, self.weight([hidden[1], output_size]), self.bias([output_size]),name="out",drop=0.8)
        #out=tf.abs(out)
        eln=self.elastic_net(lamda=0.005,alpha=0.8)

        loss = self.sse(out, label)+eln

        self.r2=self.r2(out,label)

        opt = self.adam(loss,rate=1e-3)
        sgd_opt=self.sgd(loss)

        self.x = x
        self.label = label
        self.out = out
        self.loss = loss
        self.opt = opt
        self.sgd_opt=sgd_opt

        self.bias=self.sum_bias()

        self.sess.run(tf.global_variables_initializer())


    def train_model(self,x,label):
        """
        :param x: 入力
        :param label: 　正解データ
        :return: None
        :todo : モデルの学習
        """
        train_opt=self.sess.run(self.opt,feed_dict={self.x:x,self.label:label})
        return train_opt

    def train_model_sgd(self,x,label):
        """
        :param x: 入力
        :param label: 　正解データ
        :return: None
        :todo : モデルの学習
        """
        train_opt=self.sess.run(self.sgd_opt,feed_dict={self.x:x,self.label:label})
        return train_opt


    def predict(self,x):
        """
        :param x: 入力
        :return: predict
        :todo : predictの計算
        """
        out=self.sess.run(self.out,feed_dict={self.x:x})
        return out

    def get_loss(self,x,label):
        """
        :param x: 入力
        :param label: 　正解データ
        :return: loss
        :todo : loss の計算
        """
        loss=self.sess.run(self.loss,feed_dict={self.x:x,self.label:label})
        return loss

    def save_model(self,path):
        """
        :param path: graph を保存する場所
        :return: None
        :todo : graph保存する
        """
        self.saver = tf.train.Saver()
        self.saver.save(self.sess,path)

    def get_r2(self, x, label):
        """
        :param x: 入力
        :param label: 　正解データ
        :return: loss
        :todo : loss の計算
        """
        loss = self.sess.run(self.r2, feed_dict={self.x: x,self.label: label})
        return loss

    def get_sum_bias(self):
        bias=self.sess.run(self.bias)
        return bias


def learning():

    #Ste files pathes
    label_path="../DataSets/label.npy"
    norm_label_path="../DataSets/norm_label.npy"
    train_path = "../DataSets/train_c_r099_p.npy"
    test_path= "../DataSets/test.npy"


    #Define directry to save models
    num=6
    model_path="./models/model_"+str(date.today())+"_ver02"

    model=Model()
    dataset=Datasets(label_path,train_path)


    Batch_size=32
    Epoch =dataset.get_len() // Batch_size+1
    train_num=50000
    hidden=[1000,2000,500,500,50]
    out_put=1

    model.create_model(hidden,out_put,dataset.get_size())

    #t_data=dataset.test_data()
    x =np.arange(-10,0,1.0)
    y=np.ones(len(x))
    y2 = np.ones(len(x))

    plot=RealtimePlot(x,y)

    dummy_data = np.random.random_sample( [Batch_size, dataset.get_size()])
    for i in range(train_num):
        train_loss=0

        print("Now ",i," trials were finished.")
        dataset.shuffle()
        test_data=dataset.test_data()
        batch = dataset.batch_data(Batch_size)

        #print(batch[0])
        e=0
        train_predict=[]

        #type_nan=np.isnan(batch[0])
        #train_predict = model.predict(batch[0])
        #print(train_predict)
        for _ in tqdm(range(Epoch)):


            train_opt = model.train_model(batch[0], batch[1])
            if e %300==0:
                train_loss = model.get_loss(batch[0], batch[1])
                print("   Train loss = ", train_loss)
                #is_w_nan = model.sess.run(model.weights)
                #print("Mean of weight : ",np.mean(is_w_nan))





            if Epoch % 100==0:
                #loss=model.get_loss(batch[0],batch[1])

                #print("Epoch : ",e)
                print("------------------------------------------------------")
                #print("train loss : ", loss)
                print("------------------------------------------------------")



            #train_predict = model.predict(batch[0])
            #biases=model.get_sum_bias()
            batch = dataset.batch_data(Batch_size)
            e+=1


        test_r2=model.get_r2(test_data[0],test_data[1])
        test_loss = model.get_loss(test_data[0], test_data[1])
        test_predict=model.predict(test_data[0])
        print("test R2: ",test_r2)
        print("test loss: ", test_loss, "\n")
        #print(test_predict)
        print("predict : ",np.mean(test_predict))
        #print("label : ", batch[1])


        #x.append(1)
        #y.append(test_loss)
        x=np.append(x,i)
        y=np.append(y,test_r2)
        y2=np.append(y2,test_loss)
        plot.plot_fig(y,y2)
        plot.show_fig()

        if i % 2000==0:
            model.save_model(model_path+"/model")
            plt.savefig(model_path + "/figure.png")

    plt.savefig(model_path + "/figure.png")
    model.save_model(model_path+"/model")
    plt.plot()


if __name__=="__main__":
    learning()


