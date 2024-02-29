import numpy as np
import mnist
X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")

import matplotlib.pyplot as plt
from pylab import cm
import itertools
import normaldistribution
import sigmoidfunction

gyou = 28
retsu = 28
imagecount = 10000 #画像数
classcount = 10    #ラベルが0から9までの10個
middlenodes = 50  #中間ノード数

npz = np.load('np_savez3.npz')
W1 = npz['arr_0']
W2 = npz['arr_1']
b1 = npz['arr_2']
b2 = npz['arr_3']

total = 0

for idx in range(imagecount):

    x  = (np.array(list(itertools.chain.from_iterable(X[idx])))).reshape(-1,1)

    y1 = sigmoidfunction.sigmoid(W1 @ x + b1)

    a  = W2 @ np.array(y1) + b2

    alpha = max(a)
    y2 = np.exp(a - alpha) / np.sum(np.exp(a - alpha))

    y3 = y2.tolist()
    number = y3.index(max(y3))

    #print(number, Y[idx])
    if number == Y[idx]:
        total = total + 1
accuracy = (total / imagecount) * 100
print(accuracy)
