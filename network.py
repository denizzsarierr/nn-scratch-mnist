


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


    