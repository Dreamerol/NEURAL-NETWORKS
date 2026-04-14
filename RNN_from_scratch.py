import numpy as np
import random

def binary_to_decimal(bin_list):   
    sum = 0
    for i in range(len(bin_list)):
        sum += bin_list[i]*2**(len(bin_list) - 1-i)
    return sum
    
    
print(binary_to_decimal([0,1,0]))

def dataset(num):
    bin_len = 6
    X = np.zeros([num, bin_len])
    Y = np.zeros(bin_len)
    
    for i in range(num):
        X[i] = np.array([random.randint(0, 1) for _ in range(num)])
        Y[i] = binary_to_decimal(X[i])
        return X, Y

X_train, Y_train = dataset(6)
X_test, Y_test = dataset(5)
print(X_train)
print(Y_train)
no_samples = 5
'''
RNN formula for computation
sk = xk.W_zero + sk-1.W_one
keeps the previos state
'''
class RNN:
    def __init__(self):
        self.W = [1,1]
        self.W_delta = [0.005, 0.005]
        self.W_sign = [0,0]
        
        '''
        backpropagation
        eta_p - increase step
        eta_n - decrease step
        '''
        self.eta_p = 0.005
        self.eta_n = 0.6
        
    def state(self, xk,sk):
        '''
        xk - previous state
        sk - current state
        we compute the next state based on the previous and current one
        '''
        return xk*self.W[0] + sk*self.W[1]
        
    def forward_states(self, X):
        S = np.zeros((X.shape[0], X.shape[1]+1))
        '''
        +one , because of the initial state so the zeroth column
        is yhe initial state - s
        '''
        for k in range(0, X.shape[1]):
            '''S[:,k] - means get all the els 
            from k-th colun
            '''
            next_state = self.state(X[:,k], S[:,k])
            S[:, k+1] = next_state
        return S
        
    def output_gradient(self, guess, real):
        return 2 * (guess - real)/no_samples
        
    def backward_gradient(self, X,S, grad_out):
        grad_over_time = np.zeros((X.shape[0], X.shape[1]+1))
        grad_over_time[:, -1] = grad_out
        
        wx_grad = 0
        ws_grad = 0
        
        for k in range(X.shape[1], 0, -1):
            '''
            computing dL/DW_zero which is the sum of all kth derivs by W_zero
            we have L = xk.W_zero + W_one. sk-1 so the deriv by W_zero will
            be xk, so the idea is f(g(x))' is f'(g(x)).g', so we multiply the already
            computed gardient by the g' of W_zero which in our case is xk-1
            Analogically for the deriv by W_one - which gives us sk-1
            '''
            wx_grad += np.sum(grad_over_time[:, k] * X[:, k-1])
            ws_grad += np.sum(grad_over_time[:, k] * S[:, k-1])
            
            grad_over_time[:, k-1] = grad_over_time[:, k]*self.W[1]
            '''
            here we are computing the contribution of the weights to the errors
            W_one is the weight we multiply the prev layer, that's why we multiply by it
            '''
        return (wx_grad, ws_grad), grad_over_time
    
    def update_rprop(self, X,Y, W_prev_sign, W_delta):
        S = self.forward_states(X)
        grad_out = self.output_gradient(S[:, -1], Y)
        W_grads,_ = self.backward_gradient(X,S,grad_out)
        self.W_sign = np.sign(W_grads)
        
        for i,_ in enumerate(self.W):
            if self.W_sign[i] == W_prev_sign[i]:
                W_delta[i] *= self.eta_p
            else:
                W_delta[i] *= self.eta_n
        self.W_delta = W_delta
        
    def train(self, X, Y, training_epochs):
        for ep in range(training_epochs):
            self.update_rprop(X,Y,self.W_sign, self.W_delta)
            
            for i,_ in enumerate(self.W):
                self.W[i] -= self.W_sign[i] * self.W_delta[i]
    '''
    so we keep the sign /np.sign -1, +1, based on the sign
    we update each weight based on the previous signs and weight deltas
    '''
    
rnn = RNN()
rnn.train(X_train, Y_train, 500)
print "Weight: \t", rnn.W
print "Predicted: \t", rnn.forward_states(X_test)[:, -1]
