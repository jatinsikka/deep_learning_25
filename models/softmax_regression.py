# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => ReLU activation => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        
        # Forward pass: fcl -> ReLU -> Softmax
        # Step 1: fully connected layer
        z1 = np.dot(X, self.weights['W1'])  # Shape: (N, num_classes)
        
        # Step 2: ReLU activation
        a1 = self.ReLU(z1)  # Store for backward pass
        
        # Step 3: Softmax to get probabilities
        prob = self.softmax(a1)
        
        # Step 4: Compute loss and accuracy
        loss = self.cross_entropy_loss(prob, y)
        accuracy = self.compute_accuracy(prob, y)

        # It should be around -log_e(1/num_classes) = -log_e(0.1) = log_e(10) = 2.3026 
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################

        # Backward pass: compute gradients using chain rule
        N = X.shape[0]  # batch size
        
        # Step 1: Gradient from cross-entropy + softmax
        # For softmax + cross-entropy, the gradient is: (prob - one_hot_labels)
        dL_da1 = prob.copy()
        dL_da1[np.arange(N), y] -= 1  # Subtract 1 from correct class positions
        dL_da1 /= N  # Average over batch
        
        # Step 2: Gradient through ReLU activation
        dL_dz1 = dL_da1 * self.ReLU_dev(z1)  # Element-wise multiplication
        
        # Step 3: Gradient with respect to weights W1
        # dL/dW1 = X^T @ dL/dz1
        self.gradients['W1'] = np.dot(X.T, dL_dz1)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy





        


