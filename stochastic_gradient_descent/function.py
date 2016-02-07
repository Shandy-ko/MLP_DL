# coding: utf-8
import numpy as np

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

  # print("x=%s"%x)
  # print("w=%s"%w)
  # print("b=%s"%b)
  # print("np.dot(x,w)=%s"%np.dot(x,w))
  # i=np.dot(x,w)+b
  # print("np.dot(x,w)+b=%s"%i)
  # print("sigmoid=%s"%sigmoid(np.dot(x,w)+b))
  return sigmoid(np.dot(x,w)+b)
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
  # print("x=%s"%x)
  # print("d=%s"%d)
  # print("error=%s"%error)
  # print("x.T=%s"%x.T)
  # i=x.T*error
  # print("x.T*error=%s"%i)
  w_grad = -np.mean(x.T*error, axis=1)
  #mean Compute the arithmetic mean along the specified axis.,average
  #.T transpose()
  b_grad = -np.mean(error)
  # print("--------------")
  # print("w_grad=%s"%w_grad)
  # print("--------------")
  # print("b_grad=%s"%b_grad)
  return w_grad, b_grad


def GD(x,d,w,b,e, eta=0.10, iteration=700):
  '''
  Function: GD
  Summary: 勾配法
  Attributes:
      @param (x):データ
      @param (d):ラベル
      @param (w):重み
      @param (b):バイアス
      @param (e):誤差を保存
      @param (eta) default=0.10: 学習係数
      @param (iteration) default=700: イテレーション
  Returns: 誤差，重み，バイアス
  '''
  for _ in range(iteration):
    w_grad, b_grad = grad(x,d,w,b)
    w -= eta*w_grad
    b -= eta*b_grad
    e.append(np.mean(np.abs(d-p_y_given_x(x,w,b))))
    #abs 絶対値
  return e,w,b


def SGD(x,d,w,b,e, eta=0.10, iteration=5, minibatch_size=10):
    '''
    Function: SGD
    Summary: 確率的勾配法 + ミニバッチ
    Attributes:
        @param (x):データ
        @param (d):ラベル
        @param (w):重み
        @param (b):バイアス
        @param (e):誤差を保存
        @param (eta) default=0.10: 学習係数
        @param (iteration) default=5: イテレーション
        @param (minibatch_size) default=10: ミニバッチのサイズ
    Returns: 誤差，重み，バイアス
    '''
    for _ in range(iteration):
        for index in range(0, x.shape[0], minibatch_size):
            _x = x[index:index+minibatch_size]
            _d = d[index:index+minibatch_size]
            w_grad, b_grad = grad(_x,_d,w,b)
            w -= eta*w_grad
            b -= eta*b_grad
            e.append(np.mean(np.abs(d-p_y_given_x(x, w, b))))
            #absolute Calculate the absolute value element-wise.
    return e, w, b


def SGD_momentum(x, d, w, b, e, eta=0.10, mu=0.65, iteration=50, minibatch_size=10):
    '''
    Function: SGD_momentum
    Summary: 確率的勾配法 + ミニバッチ + モメンタム
    Attributes:
        @param (x):データ
        @param (d):ラベル
        @param (w):重み
        @param (b):バイアス
        @param (e):誤差を保存
        @param (eta) default=0.10: 誤差を保存
        @param (mu) default=0.65: 係数
        @param (iteration) default=50: イテレーション
        @param (minibatch_size) default=10: ミニバッチのサイズ
    Returns: 誤差, 重み, バイアス
    '''
    wlist, blist = [w], [b]
    def momentum(mu, list):
        return mu * (list[1] - list[0])
    for _ in range(iteration):
        for index in range(0, x.shape[0], minibatch_size):
            _x = x[index:index + minibatch_size]
            _d = d[index:index + minibatch_size]
            w_grad, b_grad = grad(_x, _d, w, b)

            if len(wlist) > 1:
                w -= eta * w_grad + momentum(mu, wlist)
                b -= eta * b_grad + momentum(mu, blist)
                wlist.pop(0)
                blist.pop(0)
            else:
                w -= eta * w_grad
                b -= eta * b_grad
            wlist.append(w)
            blist.append(b)
            e.append(np.mean(np.abs(d - p_y_given_x(x, w, b))))
    return e, w, b

def SGD_adagrad(x, d, w, b, e, eta=0.10, iteration=50, minibatch_size=10):
    '''
    Function: SGD_adagrad
    Summary: 確率的勾配法 + ミニバッチ + アダグラッド
    Attributes:
        @param (x):データ
        @param (d):ラベル
        @param (w):重み
        @param (b):バイアス
        @param (e):誤差を保存
        @param (eta) default=0.10: 学習係数
        @param (iteration) default=50: イテレーション
        @param (minibatch_size) default=10: ミニバッチのサイズ
    Returns: 誤差, 重み, バイアス
    '''
    wgrad2sum = np.zeros(x.shape[1])
    bgrad2sum = 0
    for _ in range(iteration):
        for index in range(0, x.shape[0], minibatch_size):
            _x = x[index:index + minibatch_size]
            _d = d[index:index + minibatch_size]
            w_grad, b_grad = grad(_x, _d, w, b)
            wgrad2sum += np.power(w_grad, 2)
            bgrad2sum += np.power(b_grad, 2)
            w -= (eta/np.sqrt(wgrad2sum)) * w_grad
            b -= (eta/np.sqrt(bgrad2sum)) * b_grad
            e.append(np.mean(np.abs(d - p_y_given_x(x, w, b))))
    return e, w, b
