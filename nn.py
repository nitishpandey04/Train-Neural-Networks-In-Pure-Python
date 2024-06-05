from core import Value
import random

class Neuron():
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, xs):
        act = sum((wi * xi for wi, xi in zip(self.w, xs)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
    

class Layer():
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

    
class MLP():
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params
    
    def train(self, xs, ys, epochs=1, learning_rate=0.05):
        
        for i, epoch in enumerate(range(epochs)):
            
            # forward pass
            ypred = [self(x) for x in xs]
            loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

            # backward pass
            for param in self.parameters():
                param.grad = 0.0
            loss.backward()

            # update
            for param in self.parameters():
                param.data += -learning_rate * param.grad

            print(f"epoch {i + 1} | loss {round(loss.data, 2)}")
            
    def predict(self, xs):
        ypred = []
        for x in xs:
            yout = self(x).data
            ypred.append(yout)
        return ypred
    
    def evaluate(self, xs, ylabels):
        ypreds = self.predict(xs)
        rmse = pow(sum((ylabel - ypred)**2 for ylabel, ypred in zip(ylabels, ypreds)) / len(ylabels), 0.5)
        return f"rmse {round(rmse, 2)}"