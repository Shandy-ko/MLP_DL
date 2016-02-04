# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import function as func

def main():
  '''
  Function: main
  Summary: メイン関数
  '''

  m, N=2, 10
  x1, x2=np.random.randn(N,m), np.random.randn(N,m)+np.array([5,5])
  #randn 標準正規分布による (N * m) の行列
  x=np.vstack((x1,x2))
  #vstack Stack arrays in sequence vertically (row wise).
  d1, d2=np.zeros(N), np.ones(N)
  #zeros Return a new array of given shape and type, filled with zeros.
  #ones Return a new array of given shape and type, filled with ones.
  d=np.hstack((d1,d2))
  #hstack Stack arrays in sequence horizontally (column wise).
  dataset=np.column_stack((x,d))
  #colomn_stack Stack arrays in sequence horizontally (column wise), just like with hstack.
  np.random.shuffle(dataset)
  print("dataset=\n%s"%dataset)

  x, d=dataset[:,:2], dataset[:,2]
  w, b=np.random.rand(m), np.random.random()
  print("x=%s"%x)
  print("--------------")
  print("d1=%s"%d1)
  print("--------------")
  print("d2=%s"%d2)
  print("--------------")
  print("d=%s"%d)
  print("--------------")
  print("w=%s"%w)
  print("--------------")
  print("b=%s"%b)
  # e, w, b = func.GD(x, d, w, b, list())
  e, w, b = func.SGD(x, d, w, b, list())
  # e, w, b = func.SGD_momentum(x, d, w, b, list())
  # e, w, b = func.SGD_adagrad(x, d, w, b, list())
  plot(x, d, x1, x2, e, w, b)

def plot(x,d,x1,x2,e,w,b):
  '''
  Function: plot
  Summary: 描画
  Attributes:
      @param (x):データ
      @param (d):ラベル
      @param (x1):グループgのデータ
      @param (x2):グループrのデータ
      @param (e):誤差
      @param (w):重み
      @param (b):バイアス
  '''
  print np.mean(np.abs(d-func.p_y_given_x(x,w,b)))
    #mean average
  # plt.plot(e)
  # plt.show()
  # bx=np.arange(-6,10,0.1)
  # by= -b/w[1] - w[0]/[1]*bx
  # plt.xlim([-5,10])
  # plt.ylim([-5,9])
  # plt.plot(bx,by)
  # plt.scatter(x1[:,0],x1[:,1],c='g')
  # plt.scatter(x1[:,0],x1[:,1],c='r')
  # plt.show

if __name__=="__main__":
  main()
