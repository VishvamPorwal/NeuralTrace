# This file contains the engine for the neural network
import math

class NeuralValue:
    def __init__(self, value, _children = (), _op = ''):
        self.value = value
        self.grad = 0.0
        self._backward  = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, NeuralValue) else NeuralValue(other)
        out = NeuralValue(self.value + other.value, (self, other), '+')
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, NeuralValue) else NeuralValue(other)
        out = NeuralValue(self.value * other.value, (self, other), '*')
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def exp(self):
        out = NeuralValue(math.exp(self.value), (self,), 'exp')
        def _backward():
            self.grad += math.exp(self.value) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)
                          ), "supports int/float powers only"
        out = NeuralValue(self.value**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.value**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def __truediv__(self, other):  
        return self * other**-1

    def __rtruediv__(self, other):  
        return other * self**-1

    def __sub__(self, other): 
        return self + -1 * other
    
    def __rsub__(self, other): 
        return other + -1 * self

    def tanh(self):
        x = self.value
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = NeuralValue(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"NeuralValue({self.value})"
