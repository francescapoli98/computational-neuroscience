''' '''
import numpy as np

''' The `HebbianLearning` class contains methods for implementing Hebbian, Oja and Subtractive Normalization learning rules in a
neural network. '''
class HebbianLearning:
    def __init__(self, **kwargs):
        self.w_init = np.random.uniform(low=-1, high=1, size=(2,))
        self.lr= 0.001
        self.epochs=200

    def hebbian_rule(self, u, v, w):
        w = w + (self.lr * u * v) 
        return w
    
    def oja_rule(self, u, v, w, alpha):
        w = (v * u) - alpha * (np.power(v, 2) * w)
        return w
    
    # def sub_norm(self, u, v, w, alpha):
    #     w = v * u - (v * (self.one.T @ u) * self.one) / 2
    #     return w
    
    
    # def w_training(self,data):
    #     w_set=[]
    #     for i in range(epochs):
    #     data = data[:, np.random.permutation(data.shape[1])]
    #     w_old = w
    #     for j in range(data.shape[1]):
    #         u=data[:,j]
    #         v=np.dot(u,w) 
    #         w = hebbian_rule(u, v, w)
    #         #plot
    #         w_set.append(w)
    #     if linalg.norm(w-w_old) < threshold:
    #         print("Number of epochs runned:", i+1)
    #         return w_set 
    #         #break
    #     return w_set
        
    
    #def __call__():
        




'''This class in Python contains methods for calculating principal components and plotting eigenvectors.'''
class Plots:
    def __init__(self, **kwargs):
        finalW=self.final_w
        
    
    def princ_comp(self, dset):
        # Calculate the input correlation matrix
        Q = np.cov(dset.T)

        # Compute the principal eigenvector of Q
        eignvalues, eigenvectors = np.linalg.eig(Q)
        principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
        return Q, principal_eigenvector
        
    
    def plot_eig(self, **kwargs):
        Q, pc=princ_comp.Q, princ_comp.principal_eigenvector 
        # Create a figure and a set of subplots
        fig, axs = plt.subplots(3)

        # Plot data on each subplot
        axs[0].plt.scatter(dset[:, 0], dset[:, 1])#, label='Training Data')
        axs[0].set_title('Training data points')

        axs[1].plt.quiver(0, 0, finalW[0], finalW[1], angles='xy', scale_units='xy', scale=1, color='r')#, label='Final Weight Vector')
        axs[1].set_title('Final weight vector')

        axs[2].plt.quiver(0, 0, pc[0], pc[1], angles='xy', scale_units='xy', scale=1, color='g')#, label='Principal Eigenvector')
        axs[2].set_title('Principal Eigenvector')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable='box')

        # Show the plot
        fig.show()
