import numpy as np
import matplotlib.pyplot as plt
#入力 N　出力 x

def normaldistribute(x,y,N):  #行列の縦　行列の横　手前の層のノード数
 np.random.seed(seed=32)
 rnd = np.random.normal(loc=0.0, scale=np.sqrt(1 / N), size=x * y)
 rnd2 = rnd.reshape(x, y)  #二次元配列にする
 return rnd2
