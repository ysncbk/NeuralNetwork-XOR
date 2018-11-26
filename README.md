# NeuralNetwork-XOR
Neural Network scratyh coding for XOR problem

Data and learning task. The task is to train an MLP on the exclusive-OR function
XOR: {0, 1} 2 → {0, 1} given by XOR(0, 0) = XOR(1, 1) = 0; XOR(1, 0) = XOR(0,
1) = 1 with an MLP that has 2 input units and one output unit. The training data
consists of the four input-output pairs that define the XOR function.

a) The task is to train an MLP on the exclusive-OR function XOR: {0, 1}2 → {0, 1} given by XOR(0, 0) = XOR(1, 1) = 0; XOR(1, 0) = XOR(0,1) = 1 with an MLP that has 2 input units and one output unit.

Neural network structure:

For this problem we determined the following neural-network structure, three (3) input neurons (one of them is a bias equal to 1), three (3) neurons in the hidden layer (one of the is a bias equal to 1) and one (1) neuron in the output, using sigmoid function only in the layer 1/hidden layer. We used the mean squared error as the loss function:

Set up for neural network: Basically, the problem is finding the weights (w) parameters that minimize the mean Loss function (over the entire train examples). Due to the dimension of the problem [nine (9) different parameters] this can be a very evasive task. In order to deal with all the possible solution, after setting the detailed arrangement, we determined an initial random value for every w between -0.1 and 0.1. Later we ran 12,000 epochs with a learning rate of 0.15 and backpropagation algorithm in order to determine the optimum w.

b) Instead of the Boolean XOR, train a continuous version of the same problem, namely a function f: [-1, 1]2 →{0,1}, defined by f(x, y) = sign(xy). Display the performance of your final MLP by a nice 2-dim surface graphic.

Train data: For this problem, we generated 1200 data points, distributed in four quadrants (Image 4). The observed output for 1st and 3rd quadrant is zero (0) and for 2nd and 4th quadrant one (1), known output for continuous XOR function.

Neural network structure data: At the beginning we tried to use the same structure for the neural network as the first task, but it wasn´t able to predict the output with an acceptable mean squared error. After discussing with Prof. Jaeger, we added four more neurons in the hidden layer. In other words, our final model consists of three (3) input neurons (one of them is a bias equal to 1), seven (7) neurons in the hidden layer (one of the is a bias equal to 1) and one (1) neuron in the output, using sigmoid function only in the hidden layer . We used the mean squared error as the loss function:
