import numpy as np
import math

 #slicing matrix to compute the new 'sharpened' image using filters
class Convolutional:
    def __init__(self, number_of_filters):
        self.number_of_filters = number_of_filters
        self.biases = np.zeros(self.number_of_filters)
        self.filters = np.random.randn(self.number_of_filters, 3, 3)
    
    def generate_regions(self, image):
        h, w = image.shape  # FIX: shape is attribute, not method (no parentheses)
        segments = []
        for i in range(h-2):
            for j in range(w-2):
                segments.append(image[i:i+3, j:j+3])
                
        return np.array(segments)
    
    def forward(self, input):
        self.last_input = input
        self.output = []
        h, w = input.shape
        
        i = 0
        for filter in self.filters:
            result = []
            for segm in self.generate_regions(input):  # FIX: self.generate_regions
                result.append(np.dot(segm, filter) + self.biases[i])
            result = np.array(result).reshape(h-2, w-2)  # FIX: np.array() before reshape
            self.output.append(result)
            i += 1
        self.output = np.array(self.output)
        return self.output
     #for each filter there is a different output
    def backpropagation(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)  # FIX: initialize d_L_d_filters
        
        for idx, im_region in enumerate(self.generate_regions(self.last_input)):  # FIX: iterate properly
            i = idx // (self.last_input.shape[0]-2)
            j = idx % (self.last_input.shape[1]-2)
            for f in range(self.number_of_filters):  # FIX: range() instead of len()
                 #for each filter we calculate the derivative so we get the value of the deriv in [i,j] spot
                d_L_d_filters[f] += im_region * d_L_d_out[i, j, f]
        self.filters -= d_L_d_filters * learn_rate
        
        return None


def max_func(ar):
    maxi = -5555
    for l in ar:
        if l > maxi:
            maxi = l
    return maxi
    
def min_func(ar):
    mini = 555555
    for l in ar:
        if l < mini:
            mini = l
    return mini
    
def av_func(ar):
    return math.floor(sum(ar)/len(ar))


# converting the matrix, getting the max/min or avg value of each 2 by 2 submatrix 
class Pooling:
    def __init__(self, pooling_func):
        self.pooling_func = pooling_func
        
    def generate_regions(self, input):
        h, w = input.shape
        
        h_iter = h // 2
        w_iter = w // 2
        
        result = []  # FIX: initialize result before the loop
        
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                result.append(input[i:i+2, j:j+2])
        
        result = np.array(result)
        result = result.reshape(h_iter, w_iter)
        return result
                
    def forward(self, input):
        self.last_input = input
        self.output = []
        for layer in self.last_input:
            result = []
            for ar in self.generate_regions(layer):  # FIX: self.generate_regions
                result.append(self.pooling_func(ar.reshape(1,4)))
            self.output.append(np.array(result))
        self.output = np.array(self.output)
        return self.output
    
    
    def backpropagation(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        for im_region, i, j in self.generate_regions(self.last_input):  # FIX: self.last_input (was self.las_input)
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))
            for ii in range(h):
                for ji in range(w):
                    for fi in range(f):
                        '''
                        Overwriting the derivative value
                        only if the pixel matches the max.min value
                        '''
                        if im_region[ii, ji, fi] == amax[fi]:
                            
                            d_L_d_input[i* 2 + ii, j*2 + ji, fi] = d_L_d_out[i, j, fi]
        return d_L_d_input
        

class Softmax:
    def __init__(self, input_shape, num_classes):
        h, w, d = input_shape
        self.last_input_shape = input_shape
        dim = h * w * d
        self.num_classes = num_classes
        self.weights = np.random.randn(dim, num_classes) / num_classes  # FIX: / num_classes, not / len(num_classes)
        self.biases = np.zeros(num_classes)
        
    def forward(self, input):
        input = input.flatten()
        self.last_input = input
        self.last_input_shape_flat = input.shape
        self.output = np.dot(input, self.weights)
        suma = sum([np.exp(item) for item in self.output])
        self.probabilities = np.array([np.exp(item)/suma for item in self.output])
        
        return self.probabilities
   

    def backprop(self, label, learn_rate):
        num_classes = len(self.probabilities)
        d_L_d_out = np.zeros(num_classes)
        d_L_d_out[label] = -1 / self.probabilities[label]
        all_exp = np.array([np.exp(item) for item in self.probabilities])
        S = sum(all_exp)
        d_exp = -all_exp[label] * all_exp / S**2 #first case
        d_exp[label] = all_exp[label] * (S - all_exp[label]) / S**2 #second case
        '''
        Computing the derivatives -> we get -1 /yi from binary-cross entropy func -logyi
        then only the labeled idx makes the differnce and we compute
        only that derivative called d_exp ->d/dai from e^ak/sum of all exps from Softmax
        if k=i second case otherwise first case
         da_db = 1
         d_a_d_w = self.last_input
         d_a_d_inputs = self.weights
         '''

        '''
        then we get the inner derivatives
        by weight we get the input which we have stored and by input - x -> which are the weights
        '''
        
        d_L_d_w = self.last_input[:, None] @ d_exp[None, :]
        d_L_d_input = self.weights @ d_exp

        self.weights -= learn_rate * d_L_d_w
        self.biases -= learn_rate * 1

        return d_L_d_input.reshape(self.last_input_shape_flat)
