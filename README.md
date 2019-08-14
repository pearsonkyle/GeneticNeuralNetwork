# Trigomatic
A submission for the very first [/r/ProgrammerHumor hackathon, Over Engineered](https://www.reddit.com/r/ProgrammerHumor/comments/ckhow3/the_very_first_programmerhumor_hackathon_is_now/) . In the future when humankind forgets the fundamentals of mathematics artificial intelligence will be used to calculate it for you. A custom genetic algorithm is used to optimize the architecture of a neural network such that it can accurately calculate basic trig functions for you.

## Requirements
`pip install -r requirements.txt` 
- Numpy
- Matplotlib
- TensorFlow (version > 2.0)

## Training Data
Simulated data is used to train the neural network and custom data can be created using the function below
```python
def create_data(func, NUM=10000):
    X = np.random.choice( np.linspace(0,2*np.pi, NUM+1000), NUM, replace=False)
    y = func(X)
    return X.reshape(-1,1), y.reshape(-1,1)
```
Call the function like such: `create_data( np.cos, 10000)`

If you want to use a custom function, feel free but remember to change the range in which the function is evaluated. For the example above the data is evalulated between 0 and 2 pi, with 10000 random points inbetween. 

## Neural Network Architecture
TensorFlow is used to create a deep neural network that is eventually trained to compute a trig function. The class `individual` has properties that pertain to building a parameterized machine learning model, creating a random architecture and breeding/swapping traits between models for the genetic algorithm optimization. The neural network architecture is parameterized using a cos function: 
```python
layer_func = lambda x, A,w: (A*np.cos(w*np.linspace(0,np.pi/2,x))).astype(int)
```
where *x* corresponds to the number of layers, *A* is the amplitude of the first layer with all subsequent layers being smaller than that and *w* changes the rate of neural degradation. 50 random samples of the function will looks like this:

![](NN_parameterization.png)

Each network has an input size of 1 corresponding to an angle and an output size of 1 corresponding to the respective tangent function evaluation for that angle. 

## Examples

```
python genetic_network --train 
```

## Future Applications
While this program was made as a joke, the optimization using a genetic algorithm is something that can be used in modern day research particularly if you can train a neural network in a reasonable amount of time. Then you can leverage this ensamble sampling technique to find the best architecture. 

I also wonder if a binary neural network could be optimized and ultimately replace the calculations a computer makes to compute trig functions on the lowest level? Like a neural network that just does bit operations 