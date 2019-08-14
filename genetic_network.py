import argparse

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers
import tensorflow as tf

# parameterized layer architecture
layer_func = lambda x, A,w: (A*np.cos(w*np.linspace(0,np.pi/2,x))).astype(int)

class individual():
    def __init__(self,layer_sizes=[3,2], batch_size=32, lr=0.1, momentum=0.25, decay=0.001):
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.build_model()

    def build_model(self):
        inputlayer = tf.keras.Input(shape=(1), name="input")
        layer = layers.Dense(self.layer_sizes[0], activation= 'relu')(inputlayer)
        for i in range(1,len(self.layer_sizes)):
            if self.layer_sizes[i] > 0:
                layer = layers.Dense(self.layer_sizes[i], activation= 'relu')(layer)
            else:
                break
            # TODO add dropout 
        output = layers.Dense(1, activation= 'relu')(layer)
        self.model = tf.keras.Model(inputs=inputlayer, outputs=output)
        SGDsolver = tf.keras.optimizers.SGD(lr=self.lr, momentum=self.momentum, decay=self.decay, nesterov=True)
        self.model.compile(loss='mse', optimizer=SGDsolver)

    def fit(self,X,y,epochs=10):
        self.model.fit(X,y, batch_size=self.batch_size, epochs=epochs, validation_split=0.1)

    @staticmethod 
    def randomize():
        traits = {
            'layer_sizes':layer_func(
                np.random.randint(1,10),
                np.random.randint(1,100),
                np.random.random()*2 + 0.1,
            ),
            'batch_size': np.random.randint(1,64),
            'lr':np.random.random()*0.5+0.01,
            'momentum':np.random.random()*0.5+0.01,
            'decay':np.random.random()*0.01
        }
        return individual(**traits)

    @staticmethod
    def breed( parent1, parent2, mut_rate=0.05):
        traits = ['layer_sizes', 'batch_size', 'lr', 'momentum', 'decay']
        swap_traits = np.random.choice(traits, np.random.randint(1,len(traits)), replace=False)

        traits1 = {}
        traits2 = {}
        for k in traits:
            if k in swap_traits:
                traits1[k] = getattr( parent2, k)
                traits2[k] = getattr( parent1, k)
            else:
                traits1[k] = getattr( parent1, k)
                traits2[k] = getattr( parent2, k)

        # TODO mutation 
        baby1 = individual(**traits1)
        baby2 = individual(**traits2)
        return baby1, baby2
        
def create_data(func, NUM=10000):
    X = np.random.choice( np.linspace(0,2*np.pi, NUM+1000), NUM, replace=False)
    y = func(X)
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)
    return X,y

if __name__ == "__main__":

    # input args:
        # epochs
        # init size
        # num generations
    
    # create some data 
    X_train, y_train = create_data( np.cos, 10000)
    X_test, y_test = create_data( np.cos, 10000)
    
    # create lots of neural networks
    population = []
    print("Generating initial population")
    for i in range(100):
        population.append( individual.randomize() )
    
    for j in range(5): # generation 

        for i in range(len(population)):
            population[i].train
