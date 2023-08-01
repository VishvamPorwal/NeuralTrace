# This file contains the neural network classes
import random
from neuraltrace.engine import NeuralValue

class Neuron:
    def __init__(self, no_of_in):
        self.weights = [NeuralValue(random.uniform(-1, 1)) for _ in range(no_of_in)]
        self.bias = NeuralValue(random.uniform(-1, 1))

    def __call__(self, inputs):
        act = sum([i*w for i,w in zip(inputs, self.weights)]) + self.bias
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self) -> str:
        return f"Neuron with {len(self.weights)} weights and bias"
    
class Layer:
    def __init__(self, no_of_in, no_of_out):
        self.neurons = [Neuron(no_of_in) for _ in range(no_of_out)]
    
    def __call__(self, inputs):
        outs = [n(inputs) for n in self.neurons]
        if len(outs) == 1:
            return outs[0]
        return outs
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def __repr__(self) -> str:
        return f"Layer with {len(self.neurons)} neurons"
    
class MultiLayer_Perceptron:
    def __init__(self, no_of_in, no_of_outs):
        sizes = [no_of_in] + no_of_outs
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self) -> str:
        return f"MultiLayer Perceptron with {len(self.layers) - 1} hidden layers, {len(self.layers[0].neurons)} input(s) and {len(self.layers[-1].neurons)} output(s)"
