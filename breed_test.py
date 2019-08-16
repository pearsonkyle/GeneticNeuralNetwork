from genetic_network import individual, create_data
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # create some data 
    X_train, y_train = create_data( np.cos, 1000)
    X_test, y_test = create_data( np.cos, 1000)
    
    parent1 = individual.randomize()
    parent2 = individual.randomize()
    baby1, baby2 = individual.breed(parent1, parent2)
    print(parent1.traits)
    print(parent2.traits)
    print(baby1.traits)
    print(baby2.traits)
    
    parent1.fit( X_train, y_train )
    parent2.fit( X_train, y_train )

    f,ax = plt.subplots(2)

    y_pred = parent1.model.predict( X_test ) 
    mse = np.sum( (y_pred - y_test)**2) 
    print('parent1:',mse)
    ax[0].plot(X_test,y_pred,'r.',label='Prediction'); 
    ax[0].plot(X_test,y_test,'g.',label='Truth');
    ax[0].legend(loc='best')
    ax[0].set_title('Parent 1')

    y_pred = parent2.model.predict( X_test ) 
    mse = np.sum( (y_pred - y_test)**2) 
    print('parent2:',mse)
    ax[1].plot(X_test,y_pred,'r.',label='Prediction'); 
    ax[1].plot(X_test,y_test,'g.',label='Truth');
    ax[1].legend(loc='best')
    ax[1].set_title('Parent 2')
    plt.show() 