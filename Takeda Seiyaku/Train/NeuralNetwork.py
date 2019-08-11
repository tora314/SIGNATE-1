import pickle as pick
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class NN:
    def __init__(self):
        self.weights=[]
        self.biases=[]
        self.activaes=[]
        self.losses=[]
        self.opts=[]
        self.x=[]
        self.y=[]
        self.step=np.zeros(100)
        self.sess=tf.InteractiveSession()

    def relu(self,x,w,b,drop=0.5,name=None):
        """
        :param x: type tensor 入力のtensor
        :param w: type tensor 重みのtensor
        :param b: type tensor biasのtensor
        :return: type tensor  relu　name scope の設定
        """
        i = 0
        if name ==None:
            name = "relu" + str(self.step[i])

        with tf.name_scope(name):
            relu = tf.nn.relu(tf.matmul(tf.nn.dropout(x,keep_prob=drop), w) + b,name=name)
            self.activaes.append(relu)
        self.step[i] += 1
        return relu

    def leakly_relu(self,x,w,b,drop=0.5,name=None):
        """
        :param x: type tensor 入力のtensor
        :param w: type tensor 重みのtensor
        :param b: type tensor biasのtensor
        :return: type tensor  relu　name scope の設定
        """
        i = 0
        if name ==None:
            name = "leakly_relu" + str(self.step[i])

        with tf.name_scope(name):
            relu = tf.nn.leaky_relu(tf.matmul(tf.nn.dropout(x,keep_prob=drop), w) + b,name=name)
            self.activaes.append(relu)
        self.step[i] += 1
        return relu

    def relu6(self,x,w,b,drop=0.5,name=None):

        """
        :param x: type tensor 入力のtensor
        :param w: type tensor 重みのtensor
        :param b: type tensor biasのtensor
        :return: type tensor  relu　name scope の設定
        """
        i = 15
        if name ==None:
            name = "relu66" + str(self.step[i])

        with tf.name_scope(name):
            relu = tf.nn.relu6(tf.matmul(tf.nn.dropout(x,keep_prob=drop), w) + b,name=name)
            self.activaes.append(relu)
        self.step[i] += 1
        return relu

    def elu(self,x,w,b,drop=0.5,name=None):

        """
        :param x: type tensor 入力のtensor
        :param w: type tensor 重みのtensor
        :param b: type tensor biasのtensor
        :return: type tensor exponential liner unit　name scope の設定
        """
        i = 15
        if name ==None:
            name = "elu" + str(self.step[i])

        with tf.name_scope(name):
            relu = tf.nn.elu(tf.matmul(tf.nn.dropout(x,keep_prob=drop), w) + b,name=name)
            self.activaes.append(relu)
        self.step[i] += 1
        return relu

    def sigmoid(self,x,w,b,drop=1):
        """
        :param x: type tensor 入力のtensor
        :param w: type tensor 重みのtensor
        :param b: type tensor biasのtensor
        :return: type tensor sigmoid  name scope の設定
        """
        i=1
        name="sigmoid"+str(self.step[i])
        with tf.name_scope(name):
            sigmoid = tf.nn.sigmoid(tf.matmul(tf.nn.dropout(x,keep_prob=drop), w) + b,name=name)
            self.activaes.append(sigmoid)
        self.step[i]+=1
        return sigmoid

    def softmax(self,x,w,b,drop=1,name=None):
        """
        :param x: type tensor 入力のtensor
        :param w: type tensor 重みのtensor
        :param b: type tensor biasのtensor
        :return: type tensor softmax  name scope の設定
        """
        i = 2
        if name==None:
            name = "softmax" + str(self.step[i])

        with tf.name_scope(name):
            softmax = tf.nn.softmax(tf.matmul(tf.nn.dropout(x,keep_prob=drop), w) + b,name=name)
            self.activaes.append(softmax)
        self.step[i] += 1
        return softmax

    def tanh(self,x,w,b,drop=1,name=None):
        """
        :param x: type tensor 入力のtensor
        :param w: type tensor 重みのtensor
        :param b: type tensor biasのtensor
        :return: type tensor tanh  name scope の設定
        """
        i = 3
        if name==None:
            name = "tanh" + str(self.step[i])

        with tf.name_scope(name):
            tanh = tf.nn.tanh(tf.matmul(tf.nn.dropout(x,keep_prob=drop), w) + b,name=name)
            self.activaes.append(tanh)
        self.step[i] += 1
        return tanh

    def conv2d(self,x,filter,stride,padding="SAME",drop=1,name=None):
        """
        :param x: type tensor 入力のtensor
        :param w: type tensor 重みのtensor
        :param b: type tensor biasのtensor
        :return: type tensor conv  name scope の設定
        """
        i = 16
        if name==None:
            name = "conv" + str(self.step[i])

        with tf.name_scope(name):
            conv = tf.nn.conv2d(x,filter,stride,padding=padding,name=name)
            self.activaes.append(conv)
        self.step[i] += 1
        return conv

    def conv1d(self, x, filter, stride, padding="SAME", drop=1, name=None):
        """
        :param x: type tensor 入力のtensor
        :param w: type tensor 重みのtensor
        :param b: type tensor biasのtensor
        :return: type tensor conv  name scope の設定
        """
        i = 16
        if name == None:
            name = "conv" + str(self.step[i])

        with tf.name_scope(name):
            conv = tf.nn.conv1d(x, filter, stride, padding=padding, name=name)
            self.activaes.append(conv)
        self.step[i] += 1
        return conv

    def maxpool(self,x,ksize,stride,padding="SAME",drop=1,name=None):
        """
        :param x: type tensor 入力のtensor
        :param w: type tensor 重みのtensor
        :param b: type tensor biasのtensor
        :return: type tensor maxpool  name scope の設定
        """
        i = 17
        if name==None:
            name = "maxpool" + str(self.step[i])

        with tf.name_scope(name):
            pool = tf.nn.max_pool(x,ksize=ksize,strides=stride,padding=padding,name=name)
            self.activaes.append(pool)
        self.step[i] += 1
        return pool


    def weight(self,dim,min=0,max=1,norm=False,name=None):
        """
        :param dim: type list　重みの次元　
        :return: type tensor 重みのtensor
        :todo : 重みのname scope を設定して重みを返す
        """
        i = 4
        if name==None:
            name = "weight" + str(self.step[i])

        with tf.name_scope(name):
            if norm:
                w=tf.Variable(tf.truncated_normal(dim),tf.float32,name=name)
            else:
                w = tf.Variable(tf.random_uniform(dim, minval=min, maxval=max),name=name)

            self.weights.append(w)
        self.step[i] += 1
        return w

    def bias(self,dim,min=0,max=1,zeros=False,name=None):
        """
        :param dim: type list　biasの次元　
        :return: type tensor biasのtensor
        :todo : biasのname scope を設定してbiasを返す
        """
        i = 5
        if name==None:
            name = "bias" + str(self.step[i])

        with tf.name_scope(name):
            if zeros:
                b= tf.Variable(tf.zeros(dim),name=name)
            else:
                b = tf.Variable(tf.random_uniform(dim, minval=min, maxval=max),tf.float32,name=name)

            self.biases.append(b)
        self.step[i] += 1
        return b

    def mape(self,x,label):
        """
        :param x: 　　推測データ
        :param label: 正解データ
        :return: type tensor loss
        :todo : MAPE=100/n*sum(abs((x-label)/label))
        """
        i = 6
        name = "mape" + str(self.step[i])
        with tf.name_scope(name):
            loss=tf.reduce_mean(tf.abs(tf.divide(tf.subtract(x,label),label)),name=name)
            self.losses.append(loss)
        self.step[i] += 1
        return loss

    def logloss(self,x,label):
        """
        :param x: 　　推測データ
        :param label: 正解データ
        :return: type tensor loss
        :todo : logloss=1/n*sum(label*x)
        """
        i = 7
        name = "logloss" + str(self.step[i])
        with tf.name_scope(name):
            loss =tf.losses.log_loss(label,x,name=name)
            self.losses.append(loss)
        self.step[i] += 1
        return loss

    def mse(self,x,label):
        """
        :param x: 　　推測データ
        :param label: 正解データ
        :return: type tensor loss
        :todo : 二乗平均誤差=1/n*sum((x-label)**2)
        """
        i = 8
        name = "mse" + str(self.step[i])
        with tf.name_scope(name):
            loss =tf.reduce_mean(tf.square(tf.subtract(x,label)),name=name)
            self.losses.append(loss)
        self.step[i] += 1
        return loss



    def adam(self,loss,rate=1e-3):
        """
        :param loss:type tensor loss
        :return: type tensor Adam Optimizer
        """
        i = 9
        name = "adam" + str(self.step[i])
        with tf.name_scope(name):
            opt=tf.train.AdamOptimizer(rate,name=name).minimize(loss)
            self.opts.append(opt)
        self.step[i] += 1
        return opt

    def sgd(self,loss,rate=1e-2):
        """
        :param loss:type tensor loss
        :return: type tensor Adam Optimizer
        """
        i = 17
        name = "sgd" + str(self.step[i])
        with tf.name_scope(name):
            opt=tf.train.GradientDescentOptimizer(rate,name=name).minimize(loss)
            self.opts.append(opt)
        self.step[i] += 1
        return opt


    def placeholder(self,name):
        """
        :param name: type str　name scope
        :return: type tensor placeholser
        :todo : placeholserのname scope を設定してplaceholderを返す
        """

        with tf.name_scope(name):
            x=tf.placeholder(tf.float32,name=name)
            #self.x.append(x)

        return x

    def matmul(self,x,w,b,drop=0.5,name=None):
        """
        :param x: type tensor 入力のtensor
        :param w: type tensor 重みのtensor
        :param b: type tensor biasのtensor
        :return: type tensor  relu　name scope の設定
        """
        i = 10
        name = "matmul" + str(self.step[i])
        with tf.name_scope(name):
            matmul= tf.add(tf.matmul(tf.nn.dropout(x,keep_prob=drop), w) , b,name=name)
            self.activaes.append(matmul)
        self.step[i] += 1
        return matmul

    def sse(self,x,label):
        """
        :param x: 　　推測データ
        :param label: 正解データ
        :return: type tensor loss
        :todo : 二乗誤差和=sum((lx-label)**2)
        """
        i = 11
        name = "sse" + str(self.step[i])
        with tf.name_scope(name):
            loss =tf.reduce_sum(tf.square(x-label),name=name)
            self.losses.append(loss)
        self.step[i] += 1
        return loss

    def r2(self,x,label):
        """
                :param x: 　　推測データ
                :param label: 正解データ
                :return: type tensor loss
                :todo : R2={sum((x-mean(label))**2)}/{sum((lx-label)**2)}
                """
        i = 12
        name = "r2" + str(self.step[i])
        with tf.name_scope(name):
            total_error = tf.reduce_sum(tf.square(tf.subtract(label, tf.reduce_mean(label))))
            unexplained_error = tf.reduce_sum(tf.square(tf.subtract(label, x)))
            x1=tf.cast(total_error,tf.float32)
            x2=tf.cast(unexplained_error,tf.float32)
            loss= tf.subtract(1.0,tf.div(x2, x1),name=name)
            self.losses.append(loss)
        self.step[i] += 1
        return loss

    def error_r2(self,x,label):
        """
                :param x: 　　推測データ
                :param label: 正解データ
                :return: type tensor loss
                :todo : R2={sum((x-mean(label))**2)}/{sum((lx-label)**2)}
                """
        i = 15
        name = "error_r2" + str(self.step[i])
        with tf.name_scope(name):
            total_error = tf.reduce_sum(tf.square(tf.subtract(label, tf.reduce_mean(label))))
            unexplained_error = tf.reduce_sum(tf.square(tf.subtract(label, x)))
            x1=tf.cast(total_error,tf.float32)
            x2=tf.cast(unexplained_error,tf.float32)
            loss= tf.div(x2, x1,name=name)
            self.losses.append(loss)
        self.step[i] += 1
        return loss

    def mae(self,x,label):
        """
        :param x: 　　推測データ
        :param label: 正解データ
        :return: type tensor loss
        :todo : MAPE=100/n*sum(abs((x-label)/label))
        """
        i = 13
        name = "mae" + str(self.step[i])
        with tf.name_scope(name):
            loss=tf.reduce_mean(tf.abs(tf.subtract(x,label)),name=name)
            self.losses.append(loss)
        self.step[i] += 1
        return loss

    def elastic_net(self, lamda=0.01, alpha=0.5):
        """
        :param ramda: 係数１
        :param alpha: 係数２
        :return: lamda*(alpha*w+(1-alpha)*w**2)
        """
        w1=tf.Variable(0.0,tf.float32)
        w2=tf.Variable(0.0,tf.float32)
        for w in self.weights:

            #print(w)
            #with tf.Session() as sess:
                #sess.run(tf.global_variables_initializer())
            w1 =tf.add(w1, tf.reduce_sum(tf.abs(w)))
            w2 =tf.nn.l2_loss(w)

        eln=lamda*tf.add(alpha*w1,(1-alpha)*w2)
        print(w1)
        return eln

    def tensor_stack(self,packs,axis=0):
        """
        :param packs: 連結したいtensorの入ったlist
        :return: 連結したtensor
        : todo : tensor を連結する
        """
        i = 14

        name = "stacked" + str(self.step[i])

        with tf.name_scope(name):
            stack = tf.stack(packs,axis=axis)
            #self.activaes.append(stack)
        self.step[i] += 1
        return stack

    def sum_bias(self, lamda=0.0001, alpha=0.5):
        """
        :param ramda: 係数１
        :param alpha: 係数２
        :return: lamda*(alpha*w+(1-alpha)*w**2)
        """
        bias=tf.constant(0.0)
        for b in self.biases:
            #print("Hoi")
            bias=tf.add(bias , tf.reduce_sum(b))


        return bias





class Model(NN):
    # Template model class
    def __init__(self):
        super().__init__()
        self.sess=tf.InteractiveSession()

        self.opt=None
        self.loss=None
        self.out=None
        self.label=None
        self.x=None

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


    def train_model(self,x,label):
        """
        :param x: 入力
        :param label: 　正解データ
        :return: None
        :todo : モデルの学習
        """
        train_opt=self.sess.run(self.opt,feed_dict={self.x:x,self.label:label})
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






