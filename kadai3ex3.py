import numpy as np
import mnist
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

import normaldistribution
import sigmoidfunction

val = input('バッチサイズ: ')   #何枚の画像か
batchsize = int(val)    #何枚か
val2 = input('エポック数: ')   
epoch = int(val2)
gyou = 28
retsu = 28
imagecount = 60000 #画像数
classcount = 10    #ラベルが0から9までの10個
middlenodes = 50  #中間ノード数

X=np.resize(X,(60000,784))

W1 = np.array(normaldistribution.normaldistribute(middlenodes, gyou * retsu, gyou * retsu))
b1 = np.array(normaldistribution.normaldistribute(middlenodes, 1, gyou * retsu))
W2 = np.array(normaldistribution.normaldistribute(classcount, middlenodes, middlenodes))
b2 = np.array(normaldistribution.normaldistribute(classcount, 1, middlenodes))

npz = np.load('np_savez4.npz')
W1 = npz['arr_0']
W2 = npz['arr_1']
b1 = npz['arr_2']
b2 = npz['arr_3'] 

for q in range(epoch):
    Etotal = 0
    for i in range(imagecount//batchsize):   #60000/100回ループ

        w = np.random.choice(60000, size=batchsize)  #0から59999のうちランダムに100個の数字が入った配列
        
        '''for idx in w:                                   #100枚分のxを行列にする
            x  = (np.array(list(itertools.chain.from_iterable(X[idx]))))
            xx = (x.reshape(-1, 1)).T
            XXX = np.append(XXX, xx, axis=0) '''
        
        x=X[w]

        y1 = sigmoidfunction.sigmoid(W1 @ x.T + b1)   #シグモイド関数　と　全結合層1

        a  = W2 @ np.array(y1) + b2                  #全結合層2

        '''for l in range(batchsize):                      #softmax関数
            a2 = a[:, l]     #10×1
            alpha = max(a2)
            bunbo = 0
            for i in range(classcount):
                bunbo = bunbo + np.exp(a2[i] - alpha)
   
            y2 = np.exp(a2 - alpha) / bunbo
            yy = (y2.reshape(-1, 1)).T
            YYY = np.append(YYY, yy, axis=0) '''
        
        alpha=np.max(a,axis=0)
        
        YYY=np.exp(a-alpha.reshape(1,batchsize))/np.sum(np.exp(a-alpha.reshape(1,batchsize)),axis=0).reshape(1,batchsize)
   
        '''for idx2 in w:                                  #100個分の正解のone-hot vectorを行列にする
            answer = Y[idx2]
            A = np.zeros(classcount)
            A[answer] = 1
            aa = (A.reshape(-1, 1)).T
            AAA = np.append(AAA, aa, axis=0) '''
        
        AAA=np.zeros((10,batchsize))
        AAA[Y[w],np.arange(batchsize)]=1

        e = (AAA * np.log(YYY)) * -1                 # -y*logyを一気に計算
        En = np.sum(e) / batchsize                   #クロスエントロピー誤差　100個の平均
        
        L = (YYY - AAA) / batchsize    #式12
        Enak = L  # 10×batchsize

        #全結合層(2)
        EnX2 = W2.T @ Enak  #式16
        EnW2 = Enak @ y1.T  #式17
        semiEnb2 = np.sum(Enak, axis=1)  #式18
        Enb2 = semiEnb2.reshape(-1, 1)

        #シグモイド関数
        EnXsig = EnX2 * (1 - y1) * y1  #式20
        
        #全結合層(1)
        EnX1 = W1.T @ EnXsig    #式16
        EnW1 = EnXsig @ x  #式17
        semiEnb1 = np.sum(EnXsig, axis=1)  #式18
        Enb1 = semiEnb1.reshape(-1, 1)

        W1 = W1 - 0.01 * EnW1
        W2 = W2 - 0.01 * EnW2
        b1 = b1 - 0.01 * Enb1
        b2 = b2 - 0.01 * Enb2

        Etotal = Etotal + En    #60000/100　個の和
    E = Etotal/(imagecount//batchsize)
    print(E)


   
np.savez('np_savez4', W1, W2, b1, b2)
 





