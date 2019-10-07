import  matplotlib.pyplot as plt
import  numpy as np
from time import time

class RealtimePlot:
    def __init__(self,num,fig_size=(8,8)):
        #plt.ion()
        self.fig=plt.figure(figsize=fig_size)
        self.lines=[]
        self.axes=[]
        self.step=0
        self.x = list(np.arange(-10, 0, 1.0))
        self.num=num
        self.titles=[]

    def add_fig(self,init_val,pos,title="Figure"):
        x=self.x
        self.titles.append(title)

        ax=plt.subplot2grid(self.num,pos)#self.fig.add_subplot(pos)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(title,loc="right")
        ax.grid()

        line,=ax.plot(x,init_val)
        self.lines.append(line)
        self.axes.append(ax)

    def add_val(self,vals):
        self.step+=1
        self.x.append(self.step)
        x=self.x
        self.step+=1
        for i in range(len(self.lines)):
            title = self.titles[i]
            y=vals[i]
            ax = self.axes[i]
            ax.cla()                    #################################
            line ,=ax.plot(x,y) # self.lines[i]

            boxdic = {
                "facecolor": "lightgreen",
                "edgecolor": "darkred",
                "boxstyle": "Round",
                "linewidth": 2
            }


            line.set_xdata(x)
            line.set_ydata(y)

            ax.set_xlim(np.min(x),np.max(x))
            ax.set_ylim(np.min(y), np.max(y))

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_title(title, loc="right")
            ax.grid()

            ax.text(0, 1, "test loss = " + str(y[-1]), size=10, transform=ax.transAxes, bbox=boxdic)
        #plt.draw()

    def show_fig(self):
        #plt.draw()
        plt.pause(0.0001)
        #plt.cla()
def main():
    real=RealtimePlot((2,1),fig_size=(8,8))
    vals=list(np.zeros(10))
    y1=vals.copy()
    y2=vals.copy()
    y3=vals.copy()
    real.add_fig( vals,(0,0),title="Sin")
    real.add_fig(vals,(1,0),title="Cos")
    #real.add_fig(vals,313)
    step=0.01

    t=time()
    while 1:
        t1=time()
        sin=np.sin(step)
        cos=np.cos(step)
        step+=0.01
        s_time=np.abs(t-t1)

        y1.append(sin)
        y2.append(cos)
        #y3.append(np.gradient(s_time))
        real.add_val((y1,y2))
        real.show_fig()
        #t=time()

if __name__=="__main__":
    main()

