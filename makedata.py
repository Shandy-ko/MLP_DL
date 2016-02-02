# coding:utf-8
import numpy as np
def main():
  '''
  Function: main
  Summary: メイン関数
  '''

  m, N=2, 10000
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

  x, d=dataset[:,:2], dataset[:,2]
  w, b=np.random.rand(m), np.random.random()
  print("x=%s"%x)
  print("--------------")
  print("d=%s"%d)
  print("--------------")
  print("w=%s"%w)
  print("--------------")
  print("b=%s"%b)
  return x,d,w,b

main()
