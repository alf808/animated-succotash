import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        # the weights below will be initially multiplied to vector X for forward_pass
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
        
        # the weights below will be used for input to hidden layer's contribution to error
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        # training neural network steps (including backpropagation):
        # Doing a feedforward operation. (composing a series of functions)
        # Comparing the output of the model with the desired output.
        # Calculating the error.
        # Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.
        ## backpropagation is taking the derivative at each piece
        # Use this to update the weights, and get a better model.
        # Continue this until we have a model that is good.
        
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        # prediction is a combination of matrix multiplications and sigmoid functions
        # step 1, apply matrix multiplication (see lesson 2, multilayer perceptrons)
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        # step 2, apply sigmoid function which is the self.activation_function above
        hidden_outputs =  self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        # step 1, apply matrix multiplaction
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        #final_outputs = self.activation_function(final_inputs) # signals from final output layer
        final_outputs = final_inputs # activation function of the output node is f(x)=x instead of sigmoid
        
        return final_outputs, hidden_outputs # respectively, prediction, sigmoid(hidden_inputs)

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation 
         
            Arguments
            ---------
            final_outputs: output from forward pass (prediction)
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        # taking the derivative at each piece
        # READ: Lesson 2 exercise, Implementing Backpropagation
        # TODO: Output error - Replace this value with your calculations.
        # y - y_hat
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        # TODO: Calculate the hidden layer's contribution to the error
        # matrix multiplication of error and weights
        hidden_error = np.dot(error, self.weights_hidden_to_output.T)
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        # ERROR = (y − 𝑦̂) * 𝜎′(𝑥) : sigmoid_prime(x) is equivalent to sigmoid(x)*(1-sigmoid(x))
        # from forward_pass sigmoid(x) is the hidden_outputs
        # BASED on Lesson 2, backpropagation
        output_gradient = final_outputs * (1 - final_outputs) # aka sigmoid_prime
        output_error_term = error * 1
        
        hidden_gradient = hidden_outputs * (1 - hidden_outputs) # aka sigmoid_prime        
        hidden_error_term = hidden_error * hidden_gradient
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None] # X into column vector
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None] # hidden_outputs into column vector
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        #  READ: Lesson 2 exercise, Implementing Backpropagation
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        #final_outputs = self.activation_function(final_inputs) # signals from final output layer 
        final_outputs = final_inputs # activation function of the output node is f(x)=x instead of sigmoid
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 8800
learning_rate = 0.2
hidden_nodes = 15
output_nodes = 1
