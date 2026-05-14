import numpy as np

# implementasi Adam optimizer utk backpropagation
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # dictionary utk menyimpan moving averages (moment 1 dan moment 2)
        self.m = {}
        self.v = {}
        self.t = 0 # timestep/jumlah iterasi update

    def step(self, params_dict, grads_dict):
        """
        params_dict: dictionary referensi ke bobot model (misal: {'W_rnn': rnn.W})
        grads_dict: dictionary gradien yang sesuai (misal: {'W_rnn': dW_rnn})
        """
        self.t += 1 # tambah iterasi
        
        for key in params_dict.keys():
            param = params_dict[key]
            grad = grads_dict[key]
            
            # inisialisasi momentum dengan 0 pada awal training
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
                
            # 1st & 2nd moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            
            # bias-corrected 1st & 2nd moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # update bobot
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)