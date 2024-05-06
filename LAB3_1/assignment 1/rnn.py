import torch
import torch.nn as nn

''' 
This is a PyTorch neural network model class that implements a simple RNN architecture with a
specified number of hidden layers and dimensions. 
'''
class RNN_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        This function initializes an RNN model with specified input, hidden, layer, and output
        dimensions.
        
        :param input_dim: The `input_dim` parameter in the code snippet represents the dimensionality of
        the input features or input data that will be fed into the RNN model. It defines the size of the
        input vectors that the RNN will process at each time step
        :param hidden_dim: The `hidden_dim` parameter in the code snippet represents the number of
        hidden units or neurons in the RNN (Recurrent Neural Network) model. It determines the size of
        the hidden state vector that the RNN uses to capture and represent information from the input
        sequence. Increasing the `hidden_dim`
        :param layer_dim: The `layer_dim` parameter in the code snippet you provided represents the
        number of hidden layers in the RNN (Recurrent Neural Network) model. It specifies how many
        recurrent layers are stacked on top of each other in the RNN architecture. Each hidden layer
        processes the input sequence and passes its output
        :param output_dim: The `output_dim` parameter in the code snippet you provided represents the
        dimensionality of the output of the RNN model. It specifies the number of output units in the
        final linear layer (`self.fc`) of the model. The output dimension is typically determined by the
        specific requirements of the task you are
        """
        super(RNN_model, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        This function performs a forward pass through a recurrent neural network (RNN) with a fully
        connected layer at the end.
        
        :param x: The `x` parameter in the `forward` method is typically the input data that is passed
        to the model for processing. It could be a sequence of data points, such as a time series or a
        sequence of words in natural language processing tasks. In this code snippet, it seems like `x
        :return: the output of the neural network after processing the input `x`. The input `x` is
        passed through the RNN layer with an initial hidden state of zeros, and then the final output of
        the RNN is passed through a fully connected layer (`fc`) to get the final output. This final
        output is what is being returned by the function.
        """
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out
