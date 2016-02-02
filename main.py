# coding: utf-8
import numpy as np
import makedata as md
def p_y_given_x(x,w,b):
  '''
  Function: p_y_given_x
  Summary: 写像
  Attributes:
    @param (x): データ
    @param (w): 重み
    @param (b): バイアス
  Return: 写像後の値
  '''

  def sigmoid(u):
    return 1./(1.+np.exp(-u))

  print sigmoid(np.dot(x,w)+b)
  #dot Dot product of two arrays.


def grad(x,d,w,b):
  '''
  Function: grad
  Summary: 勾配の計算
  Attributes:
    @param (x): データ
    @param (d): ラベル
    @param (w): 重み
    @param (b): バイアス
  Return: 重みの勾配の平均，バイアスの勾配の平均
  '''
  error = d-p_y_given_x(x,w,b)
  w_grad = -np.mean(x.T*error, axis=1)
  b_grad = -np.mean(error)
  print("--------------")
  print("w_grad=%s"%w_grad)
  print("--------------")
  print("b_grad=%s"%b_grad)
  return w_grad, b_grad

x,d,w,b=md.main()
grad(x,d,w,b)
