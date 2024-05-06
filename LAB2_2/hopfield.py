'''
Implement all the code that is required for the different stages in the operation of a
Hopfield network, i.e., the storage phase (learning) and the retrieval phase
(initialization, iteration until convergence, outputting).

Note: in the code you need to include the computation of the overlap functions (with
respect to the training patterns) and of the energy function. 

Use the 3 input vectors p0, p1 and p2 (from the corresponding csv files in the archive) to train the Hopfield network
'''

import numpy as np
from numpy import einsum, fill_diagonal, random


class Hopfield:
    def __init__(self, img: np.array, patterns: np.array):
        """
        This function initializes parameters for a neural network model with image data and patterns.
        
        :param img: The `img` parameter is expected to be a NumPy array representing the input data. In
        this case, it seems to be the original data that has been distorted into different patterns
        :type img: np.array
        :param patterns: The `patterns` parameter is an array containing a set of patterns. In the
        context of the code snippet you provided, it seems like these patterns are related to image data
        processing or neural network training. The `patterns` array likely contains data that represents
        specific features or characteristics that the neural network is trained
        :type patterns: np.array
        """
        self.x = img #input_data (the original p0, p1, p2 then distorted into x0, x1, x2)
        self.N =  self.x.shape[0] #number of neurons (1024)
        self.p = patterns #set of patterns
        self.epochs = 100
        self.bias = None #TO BE CHANGED
    
    
    def storage(self):
        """
        The `storage` function calculates the dot product of a matrix with its transpose, sets the diagonal
        elements to zero, and returns the resulting matrix.
        :return: The `storage` method is returning a matrix `w` that is calculated using the Einstein
        summation convention with the numpy `einsum` function. The matrix `w` is the result of multiplying
        the transpose of `self.p` with `self.p`, and then setting the diagonal elements of the resulting
        matrix to 0.
        """
        w=np.einsum('ij,ik->jk', self.p, self.p)         #w=((self.p.T@self.p).sum(axis=0))/self.N
        np.fill_diagonal(w, 0)
        return w   
    
    
    def overlap_func(self, x: np.array):
        """
        The function `overlap_func` calculates the overlap of an input array with a set of patterns.
        
        :param x: The `x` parameter is expected to be a NumPy array that you will pass to the
        `overlap_func` method. The method calculates the overlap of the input `x` with each pattern in
        `self.p` and returns a list of overlap values
        :type x: np.array
        :return: The function `overlap_func` is returning a list `o` containing the dot product of each
        pattern in `self.p` with the input array `x`, divided by the total number of elements `N` in
        each pattern.
        """
        o = [(pattern @ x / self.N) for pattern in self.p]
        return o
    
    def energy_func(self, x: np.array, w: np.array):
        """
        This function calculates the energy based on input arrays x and w using matrix multiplication.
        
        :param x: The parameter `x` is a numpy array representing the input data or features. It is
        typically a vector or matrix containing the input values for a model
        :type x: np.array
        :param w: The parameter `w` in the `energy_func` function appears to be a numpy array. It is
        used as a weight matrix in the calculation of energy. The function calculates the energy based
        on the input `x` and the weight matrix `w`
        :type w: np.array
        :return: the energy value calculated based on the input arrays `x` and `w`.
        """
        e = - x.T @ w @ x / 2 #- sum(self.bias * x)
        return e
        
        
    def __call__(self):
        """
        This function performs a stochastic update of neuron states over multiple epochs, calculating
        overlap and energy functions at each step.
        :return: three values: a list of overlaps, a list of energies, and the updated state vector x.
        """
        w = self.storage()
        x=self.x
        overlaps = []
        energies = []
        for epoch in range(self.epochs):
            indexes = np.random.permutation(self.N) #vector of random indexes for state update
            #for each epoch I calculate the overlap functions and the energy function
            #for n in x: #for each set of states correspondent to a pattern in the neurons' set
            for ind in indexes: 
                #get a random neuron and update its state through the update function
                update = np.sign(w[ind] @ x) #ADD BIAS HERE
                x[ind] = update
            #overlap
            overlap = self.overlap_func(x) 
            #energy
            energy = self.energy_func(x, w)
            overlaps.append(overlap)
            energies.append(energy)
        return(overlaps, energies, x) 