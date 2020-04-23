from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tfv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getKaggleMNIST
from LazyProgrammer_unsupervised_class2_code.autoencoder_tf import DNN

tf.disable_eager_execution()

class RBM(object):
    def __init__(self, D, M, an_id):
        self.D = D
        self.M = M
        self.id = an_id
        self.build(D,M)

    def set_session(self, session):
        self.session = session

    def build(self, D, M):
        ## params
        self.W = tf.Variable(tfv2.random.normal(shape = (D, M))*np.sqrt(2.0/M))
        ## note: without the limiting variance, you get numerical stability issues
        self.c = tf.Variable(np.zeros(M).astype(np.float32))
        self.b = tf.Variable(np.zeros(D).astype(np.float32))

        ## Data
        self.X_in = tf.placeholder(tf.float32, shape=(None, D))

        ## Conditional Probabilities
        V = self.X_in
        p_h_given_v = tf.nn.sigmoid(tf.matmul(V, self.W) + self.c)
        self.p_h_given_v = p_h_given_v

        ## Sampling numbers between 0 and 1 to approximate h
        r = tf.random_uniform(shape = tf.shape(p_h_given_v))
        H = tf.to_float(r < p_h_given_v)

        p_v_given_h = tf.nn.sigmoid(tf.matmul(H, tf.transpose(self.W)) + self.b)

        r = tf.random_uniform(shape = tf.shape(p_v_given_h))
        X_sample = tf.to_float(r < p_v_given_h)

        ## Build the objective
        objective = tf.reduce_mean(self.free_energy(self.X_in)) - tf.reduce_mean(self.free_energy(X_sample))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(objective)

        ## Build the cost function
        ## Note that we won't use this to optimize the model parameters
        ## We just want to observe what happens during training
        logits = self.forward_logits(self.X_in)
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.X_in, logits = logits))

    def fit(self, X, epochs = 1, batch_sz = 100, show_fig=False):
        N, D = X.shape
        n_batches = N//batch_sz

        costs = []
        print(f"training rbm: {self.id}")
        for i in range(epochs):
            print("epoch:",i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                _, c = self.session.run((self.train_op, self.cost), feed_dict = {self.X_in: batch})
                if j % 10 == 0:
                    print("j / n_batches:", j, "/", n_batches,"cost:", c)
                costs.append(c)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def free_energy(self, V):
        b = tf.reshape(self.b, (self.D,1))
        first_term = -tf.matmul(V,b)
        first_term = tf.reshape(first_term,(-1,))

        second_term = -tf.reduce_sum(tf.nn.softplus(tf.matmul(V,self.W) + self.c), axis = 1)

        return first_term + second_term

    def forward_hidden(self, X):
        return tf.nn.sigmoid(tf.matmul(X, self.W) + self.c)

    def forward_logits(self, X):
        Z = self.forward_hidden(X)
        return tf.matmul(Z, tf.transpose(self.W)) + self.b

    def forward_output(self, X):
        return tf.nn.sigmoid(self.forward_logits(X))

    def transform(self, X):
        ## accepts and returns a real numpy array
        ## unlike forward_hidden and forward_ouput
        ## which deal in tensorflow variables
        return self.session.run(self.p_h_given_v, feed_dict={self.X_in: X})


def main():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

    ## same as the autoencoder_tf.py file
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    _, D = Xtrain.shape
    K = len(set(Ytrain))
    dnn = DNN(D, [1000, 750, 500], K, UnsupervisedModel= RBM)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        dnn.set_session(session)
        dnn.fit(Xtrain, Ytrain, Xtest, Ytest, pretrain = True, epochs = 10)


if __name__ == '__main__':
    main()




