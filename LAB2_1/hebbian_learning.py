''' '''

class HebbianLearning:
    def __init__(self,**kwargs):
        w = random.uniform(low=-1, high=1, size=None)        

    def hebbian_rule(self, w, lr):
        w = w + (lr * u * v) 
        return w
    
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
