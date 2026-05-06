


class NeuralNetwork:


    def __init__(self,architecture,n_iters=100,learning_rate=0.01,batch_size=32):

        self.architecture = architecture
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.W = []
        self.b = []

        self.Z_cache = [0 for x in range(len(architecture) - 1)]
        self.A_cache = [0 for x in range(len(architecture) - 1)]

        self._set_parameters()

    def _set_parameters(self):

        for i in range(1,len(self.architecture)):
            
            # Xavier weight initialization for ReLU activation function
            w_current_layer = np.random.randn(self.architecture[i],self.architecture[i-1]) * np.sqrt(2 / self.architecture[i - 1])
            b_current_layer = np.zeros((1,self.architecture[i]))

            self.W.append(w_current_layer)
            self.b.append(b_current_layer)


    # ReLU activation function
    def relu_activation(self,z):

        return np.maximum(0,z)

    # Derivative for computing
    def relu_derivative(self,z):

        return np.where(z > 0, 1, 0)

    # Softmax activation for output layer
    def softmax_activation(self,z):

        # To avoid overflow
        z_shift = z - np.max(z, axis=1, keepdims=True)
        z_exp = np.exp(z_shift)

        return z_exp / np.sum(z_exp,axis = 1, keepdims=True)

    def create_batches(self,X,y):

        # Shuffling to break inherent ordering
        n_samples = X.shape[0]
        shuffled_indices = np.random.permutation(n_samples)

        X_s = X[shuffled_indices]
        y_s = y[shuffled_indices]

        for i in range(0,n_samples,self.batch_size):

            yield X_s[i:i+self.batch_size], y_s[i:i+self.batch_size]


    def forward_prop(self,X):

        A_previous = X
        
        for i in range(len(self.W)):

            Z = np.dot(A_previous,self.W[i].T) + self.b[i]

            if i == len(self.W) - 1:

                A = self.softmax_activation(Z)

            else:

                A = self.relu_activation(Z)

            A_previous = A

            self.Z_cache[i] = Z
            self.A_cache[i] = A

        return A

    def backward_prop(self,A,X,y):

        n_samples = X.shape[0]

        dz = A - y

        for i in reversed(range(len(self.W))):
            
            A_previous = X if i == 0 else self.A_cache[i-1]


            dw = (np.dot(dz.T,A_previous)) / n_samples
            db = (np.sum(dz,axis = 0,keepdims=True)) / n_samples

            # Set dZ for the next layer
            if i != 0:
                
                Z_previous = self.Z_cache[i-1]
                dA_previous = np.dot(dz,self.W[i])
                dz = dA_previous * self.relu_derivative(Z_previous)

            self.W[i] -= self.learning_rate * dw
            self.b[i] -= self.learning_rate * db
