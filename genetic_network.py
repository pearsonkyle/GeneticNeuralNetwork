import argparse

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers
import tensorflow as tf

# parameterized layer architecture
layer_func = lambda x, A,w: (A*np.cos(w*np.linspace(0,np.pi/2,x))).astype(int)

class individual():
    def __init__(self,layer_sizes=[3,2], batch_size=32, learning_rate=0.1, momentum=0.25, decay=0.001, dropout=0.1):
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.momentum = momentum
        self.dropout = dropout
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
            if i == 1:
                layer = layers.Dropout(self.dropout)(layer)
        output = layers.Dense(1, activation= 'linear')(layer)
        self.model = tf.keras.Model(inputs=inputlayer, outputs=output)
        SGDsolver = tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate, 
            momentum=self.momentum, 
            decay=self.decay, 
            nesterov=True
        )
        self.model.compile(loss='mean_squared_error', optimizer=SGDsolver)

    def fit(self,X,y,epochs=20):
        self.model.fit(X,y, batch_size=self.batch_size, epochs=epochs, verbose=False)#, validation_split=0.1)
    
    @property
    def traits(self):
        traits = ['layer_sizes', 'batch_size', 'learning_rate', 'momentum', 'decay', 'dropout']
        td = {} 
        for k in traits: td[k] = getattr(self,k)
        return td
    
    @staticmethod 
    def random_traits():
        traits = {
            'layer_sizes':layer_func(
                np.random.randint(3,10),
                np.random.randint(2,100),
                np.random.random()*2 + 0.1,
            ),
            'learning_rate': np.round( np.random.random()*0.5+0.01, 4),
            'momentum': np.round(np.random.random()*0.5+0.01, 4),
            'dropout': np.round(np.random.random()*0.25, 4),
            'decay': np.round(np.random.random()*0.0001,6),
            'batch_size': np.random.randint(1,32),
        }
        return traits

    @staticmethod 
    def randomize():
        return individual(**individual.random_traits())

    @staticmethod
    def breed( parent1, parent2, mut_rate=0.01):
        # randomize cross over
        traits = ['layer_sizes', 'batch_size', 'learning_rate', 'momentum', 'decay', 'dropout']
        swap_traits = np.random.choice(traits, np.random.randint(1,len(traits)-1), replace=False)
        traits1 = {}
        traits2 = {}
        for k in traits:
            if k in swap_traits:
                traits1[k] = getattr( parent2, k)
                traits2[k] = getattr( parent1, k)
            else:
                traits1[k] = getattr( parent1, k)
                traits2[k] = getattr( parent2, k)

        # randomly mutate baby1 
        if np.random.random() < mut_rate:
            rand_traits = individual.random_traits()
            swap_traits = np.random.choice(traits, np.random.randint(1,len(traits)-3), replace=False)
            for k in traits:
                if k in swap_traits:
                    traits1[k] = rand_traits[k]

        # mutate 2 
        if np.random.random() < mut_rate:
            rand_traits = individual.random_traits()
            swap_traits = np.random.choice(traits, np.random.randint(1,len(traits)-3), replace=False)
            for k in traits:
                if k in swap_traits:
                    traits2[k] = rand_traits[k]

        return individual(**traits1), individual(**traits2)


def create_data(func, NUM=10000):
    X = np.random.choice( np.linspace(0,2*np.pi, NUM+1000), NUM, replace=False)
    y = func(X)
    return X.reshape(-1,1), y.reshape(-1,1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    help_ = "Number of training epochs"
    parser.add_argument("-e", "--epochs", help=help_, default=10, type=int)
    help_ = "Initial population size"
    parser.add_argument("-p", "--population", help=help_, default=50, type=int)
    help_ = "Number of generations"
    parser.add_argument("-g", "--generations", help=help_, default=5, type=int)
    args = parser.parse_args()

    # create some data 
    X_train, y_train = create_data( np.cos, 1000)
    X_test, y_test = create_data( np.cos, 1000)
    
    # create lots of neural networks
    population = []
    mse = []
    print("Generating initial population")
    for i in range(args.population): 
        population.append( individual.randomize() )
    
    # loop through generations and breed
    for j in range(args.generations): 
        for i in range(len(population)):
            population[i].fit(X_train, y_train, epochs=args.epochs)

            # evaluate fitness
            y_pred = population[i].model.predict( X_test ) 
            mse.append( np.sum( (y_pred - y_test)**2) )

        idx = np.argsort(mse)
        # remove lower 50% of models 

        # breed 50% of best models randomly 
        for i in range(int(len(population)*0.5)):
            pass

        # reset weights
        population[i].model.reset_states()