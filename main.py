from modules.layer import Layer
from modules.network import Network

INPUT = [1, 2, 3]
RESULTS = [3.2, 7.4]

# create the layers
layer1 = Layer(shape=len(INPUT), size=5)
layer2 = Layer(shape=len(layer1.neurons), size=len(RESULTS))

# create the network
network = Network(layers=[layer1, layer2])
network.train(inputs=INPUT, epochs=50, expected=RESULTS, learning_rate=0.01)
