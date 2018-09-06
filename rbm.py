# python: 3.6
# encoding: utf-8

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import time

class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, n_observe=784):
        """Initialize model."""

        #self.W = np.random.random((n_hidden, n_observe))    # weights vector
        #self.vbias = np.random.random((1, n_observe))   # bias for visible nodes(a)
        #self.hbias = np.random.random((1, n_hidden))    # bias for hidden nodes(b)
        self.W = np.zeros((n_hidden, n_observe))
        self.vbias = np.zeros((1, n_observe))
        self.hbias = np.zeros((1, n_hidden))
        self.n_hidden = n_hidden
        self.n_observe = n_observe

    def train(self, data, epoch=1, lr=1e-3):
        """
        Train model using data.
        lr indicates learning rate, default is 1e-3.
        """

        N = data.shape[0]

        for j in range(epoch):
            #learning rate decay
            #lr -= 1e-4
            startTime = time.time()
            for i in range(N):
                v = data[i]
                v = v[np.newaxis, :]    # 1 * n_observe
                ph_v = 1 / (1 + np.exp(- (self.hbias + np.dot(v, np.transpose(self.W)))))   
                h = np.random.uniform(0, 1, ph_v.shape)    # 1 * n_hidden
                h -= ph_v
                h[h >= 0] = 0
                h[h < 0] = 1

                pv_h = 1 / (1 + np.exp(- (self.vbias + np.dot(h, self.W))))
                newv = np.random.uniform(0, 1, pv_h.shape) # 1 * n_observe
                newv -= pv_h
                newv[newv >= 0] = 0
                newv[newv < 0] = 1

                ph_v = 1 / (1 + np.exp(- (self.hbias + np.dot(newv, np.transpose(self.W)))))
                newh = np.random.uniform(0, 1, ph_v.shape)    # 1 * n_hidden
                newh -= ph_v
                newh[newh >= 0] = 0
                newh[newh < 0] = 1

                # update parameters
                deltaW = lr * (np.dot(np.transpose(h), v) - np.dot(np.transpose(newh), newv))
                self.W += deltaW
                deltav = lr * (v - newv)
                self.vbias += deltav
                deltah = lr * (h - newh)
                self.hbias += deltah
                if np.linalg.norm(deltaW) <= 1e-5 and np.linalg.norm(deltav) <= 1e-5 and np.linalg.norm(deltah) <= 1e-5:
                    print ("-----------------------epoch: ", j, " data: ", i, " updates convergence---------------------")
                    print ("distance deltaW: ", np.linalg.norm(deltaW))
                    print ("distance deltav: ", np.linalg.norm(deltav))
                    print ("distance deltah: ", np.linalg.norm(deltah))
            endTime = time.time()
            total = endTime - startTime
            print ("-----------------------epoch: ", j, "---------------------")
            print ("distance deltaW: ", np.linalg.norm(deltaW))
            print ("distance deltav: ", np.linalg.norm(deltav))
            print ("distance deltah: ", np.linalg.norm(deltah))
            print ("training time: ", total)
            print ("-----------------------epoch: ", j, " finished---------------------")

    def trainWithMatrix(self, data, epoch=1, lr=1e-3):
        for i in range(epoch):
            startTime = time.time()
            # matrix calculation
            V = data.copy()

            pH_V = 1 / (1 + np.exp(- (self.hbias + np.dot(V, np.transpose(self.W)))))
            H = np.random.uniform(0, 1, pH_V.shape) # N * n_hidden
            H -= pH_V
            H[H >= 0] = 0
            H[H < 0] = 1

            pV_H = 1 / (1 + np.exp(- (self.vbias + np.dot(H, self.W))))
            newV = np.random.uniform(0, 1, pV_H.shape)  # N * n_observe
            newV -= pV_H
            newV[newV >= 0] = 0
            newV[newV < 0] = 1

            pH_V = 1 / (1 + np.exp(- (self.hbias + np.dot(newV, np.transpose(self.W)))))
            newH = np.random.uniform(0, 1, pH_V.shape)  # N * n_hidden
            newH -= pH_V
            newH[newH >= 0] = 0
            newH[newH < 0] = 1

            # update
            deltaW = lr * (np.dot(np.transpose(H), V) - np.dot(np.transpose(newH), newV))
            deltaH = lr * np.sum(H - newH, axis=0)
            deltaV = lr * np.sum(V - newV, axis=0)
            self.W += deltaW
            self.hbias += deltaH
            self.vbias += deltaV

            endTime = time.time()
            total = endTime - startTime
            print ("-----------------------epoch: ", i, "---------------------")
            print ("distance deltaW: ", np.linalg.norm(deltaW))
            print ("distance deltav: ", np.linalg.norm(deltaV))
            print ("distance deltah: ", np.linalg.norm(deltaH))
            print ("training time: ", total)
            print ("-----------------------epoch: ", i, " finished---------------------")
            
    def sample(self, mnist, N=100):
        """Sample from trained model."""

        #V = np.random.randint(2, size=(N, self.n_observe))  # N * n_observe
        V = mnist[:N, :].copy()

        for i in range(N):
            pH_V = 1 / (1 + np.exp(- (self.hbias + np.dot(V, np.transpose(self.W)))))
            H = pH_V.copy() # N * n_hidden
            H[H >= 0.5] = 1
            H[H < 0.5] = 0

            pV_H = 1 / (1 + np.exp(- (self.vbias + np.dot(H, self.W))))
            V = pV_H.copy()
            V[V >= 0.5] = 1
            V[V < 0.5] = 0
        return V
    
    def GibbsSample(self, mnist, N=100, iter=100):
        V = mnist[:N, :].copy()

        for i in range(iter):
            pH_V = 1 / (1 + np.exp(- (self.hbias + np.dot(V, np.transpose(self.W)))))
            H = np.random.uniform(0, 1, pH_V.shape)
            H -= pH_V
            H[H >= 0] = 0
            H[H < 0] = 1

            pV_H = 1 / (1 + np.exp(- (self.vbias + np.dot(H, self.W))))
            V = np.random.uniform(0, 1, pV_H.shape)
            V -= pV_H
            V[V >= 0] = 0
            V[V < 0] = 1
        return V

def saveImg(data, rawData):

    n_imgs, n_rows, n_cols = data.shape

    for i in range(n_imgs):
        scipy.misc.imsave('./sampleImg/trainingCmp/' + str(i) + '.jpg', data[i])
        scipy.misc.imsave('./sampleImg/mnist' + str(i) + '.jpg', rawData[i])

# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    # load mnist dataset, no label
    mnist = np.load('mnist_bin.npy')  # 60000x28x28
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    mnist.resize((n_imgs, img_size))

    # construct rbm model
    rbm = RBM(10, img_size)

    # train rbm model using mnist
    rbm.train(mnist, epoch=20, lr=0.1)
    
    #rbm.trainWithMatrix(mnist, epoch=50, lr=1e-5)
    
    # sample from rbm model
    #s = rbm.sample(mnist, N=2)
    s = rbm.GibbsSample(mnist, N=2, iter=100)
    mnist.resize((n_imgs, n_rows, n_cols))
    n_imgs, size = s.shape
    s.resize((n_imgs, n_rows, n_cols))
    saveImg(s, mnist)
    print ("sample data shape: ", s.shape)

# parameters:
# hidden: 10; epoch: 20; lr: 0.1
