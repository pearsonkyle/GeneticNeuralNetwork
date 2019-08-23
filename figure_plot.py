import matplotlib.pyplot as plt
import numpy as np

# parameterized layer architecture
layer_func = lambda x, A,w: (A*np.cos(w*np.linspace(0,np.pi/2,x))).astype(int)

if __name__ == "__main__":
    cm = plt.cm.get_cmap('jet')
    f,ax = plt.subplots(1)
    for i in range(50):
        sizes = layer_func(
            np.random.randint(1,10),
            np.random.randint(1,100),
            np.random.random()*2 + 0.1,
        )
        for j in range(len(sizes)):
            if sizes[j] <= 0:
                break
        ax.plot(1+np.arange(len(sizes[:j])), sizes[:j], color =cm(i/50))

    ax.set_ylabel('Number of Neurons')
    ax.set_xlabel('Layer Number')
    ax.set_title('Random Samples of Parameterized Architecture')
    ax.grid(True,ls='--')
    plt.tight_layout()
    plt.savefig('NN_parameterization.png')