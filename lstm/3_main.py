# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-04-28 22:27:12
# @Last Modified by:   vamshi
# @Last Modified time: 2018-04-29 14:39:10
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


#get data
data = open("3stud.txt","r").read() 
#data = strip_headers(load_etext(2701)).strip()
chars = list(set(data))
data_size, X_size = len(data), 256
print 'data has %d characters, %d unique.' % (data_size, len(chars))

#create dictionaries
idx_to_char = {i:chr(i) for i in range(256)}
char_to_idx = {c:ord(c) for c in idx_to_char.values()}


H_size = 100 # Size of the hidden layer
T_steps = 4 # Number of time steps (length of the sequence) used for training
learning_rate = 1e-1 # Learning rate
weight_sd = 0.1 # Standard deviation of weights for initialization
z_size = H_size + X_size # Size of concatenate(H, X) vector

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y

class Param:
    def __init__(self, name, value):
        self.name = name
        self.v = value #parameter value
        self.d = np.zeros_like(value) #derivative
        self.m = np.zeros_like(value) #momentum for AdaGrad

class Parameters:
    def __init__(self):
        self.W_f = Param('W_f', 
                         np.random.uniform(-np.sqrt(1.0/X_size), np.sqrt(1.0/X_size), size=(H_size,z_size)))
        self.b_f = Param('b_f',
                         np.zeros((H_size, 1)))

        self.W_i = Param('W_i',
                         np.random.uniform(-np.sqrt(1.0/X_size), np.sqrt(1.0/X_size), size=(H_size,z_size)))
        self.b_i = Param('b_i',
                         np.zeros((H_size, 1)))

        self.W_C = Param('W_C',
                         np.random.uniform(-np.sqrt(1.0/X_size), np.sqrt(1.0/X_size), size=(H_size,z_size)))
        self.b_C = Param('b_C',
                         np.zeros((H_size, 1)))

        self.W_o = Param('W_o',
                         np.random.uniform(-np.sqrt(1.0/X_size), np.sqrt(1.0/X_size), size=(H_size,z_size)))
        self.b_o = Param('b_o',
                         np.zeros((H_size, 1)))

        #For final layer to predict the next character
        self.W_v = Param('W_v',
                         np.random.uniform(-np.sqrt(1.0/X_size), np.sqrt(1.0/X_size), size=(X_size,H_size)))
        self.b_v = Param('b_v',
                         np.zeros((X_size, 1)))
        
    def all(self):
        return [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v,
               self.b_f, self.b_i, self.b_C, self.b_o, self.b_v]
        

class LSTM:
	def __init__(self,parameters):
		self.parameters = parameters


	def forward(self,x, h_prev, C_prev):
	    
	    p = self.parameters
	    z = np.row_stack((h_prev, x))
	    f = sigmoid(np.dot(p.W_f.v, z) + p.b_f.v)
	    i = sigmoid(np.dot(p.W_i.v, z) + p.b_i.v)
	    C_bar = tanh(np.dot(p.W_C.v, z) + p.b_C.v)

	    C = f * C_prev + i * C_bar
	    o = sigmoid(np.dot(p.W_o.v, z) + p.b_o.v)
	    h = o * tanh(C)

	    v = np.dot(p.W_v.v, h) + p.b_v.v
	    y = np.exp(v) / np.sum(np.exp(v)) #softmax

	    return z, f, i, C_bar, C, o, h, v, y


	def backward(self,target, dh_next, dC_next, C_prev,
	             z, f, i, C_bar, C, o, h, v, y):
	    
	    p = self.parameters    
	    dv = np.copy(y)
	    dv[target] -= 1

	    p.W_v.d += np.dot(dv, h.T)
	    p.b_v.d += dv

	    dh = np.dot(p.W_v.v.T, dv)        
	    dh += dh_next
	    do = dh * tanh(C)
	    do = dsigmoid(o) * do
	    p.W_o.d += np.dot(do, z.T)
	    p.b_o.d += do

	    dC = np.copy(dC_next)
	    dC += dh * o * dtanh(tanh(C))
	    dC_bar = dC * i
	    dC_bar = dtanh(C_bar) * dC_bar
	    p.W_C.d += np.dot(dC_bar, z.T)
	    p.b_C.d += dC_bar

	    di = dC * C_bar
	    di = dsigmoid(i) * di
	    p.W_i.d += np.dot(di, z.T)
	    p.b_i.d += di

	    df = dC * C_prev
	    df = dsigmoid(f) * df
	    p.W_f.d += np.dot(df, z.T)
	    p.b_f.d += df

	    dz = (np.dot(p.W_f.v.T, df)
	         + np.dot(p.W_i.v.T, di)
	         + np.dot(p.W_C.v.T, dC_bar)
	         + np.dot(p.W_o.v.T, do))
	    dh_prev = dz[:H_size, :]
	    dC_prev = f * dC
	    
	    return dh_prev, dC_prev

	def clear_gradients(self):
		params = self.parameters
		for p in params.all():
			p.d.fill(0)

	def clip_gradients(self):
		params = self.parameters
		for p in params.all():
			np.clip(p.d, -1, 1, out=p.d)

	def forward_backward(self,inputs, targets, h_prev, C_prev):
	    
	    # To store the values for each time step
	    x_s, z_s, f_s, i_s,  = {}, {}, {}, {}
	    C_bar_s, C_s, o_s, h_s = {}, {}, {}, {}
	    v_s, y_s =  {}, {}
	    
	    # Values at t - 1
	    h_s[-1] = np.copy(h_prev)
	    C_s[-1] = np.copy(C_prev)
	    
	    loss = 0
	    # Loop through time steps
	    for t in range(len(inputs)):
	        x_s[t] = np.zeros((X_size, 1))
	        x_s[t][inputs[t]] = 1 # Input character
	        
	        (z_s[t], f_s[t], i_s[t],
	        C_bar_s[t], C_s[t], o_s[t], h_s[t],
	        v_s[t], y_s[t]) = \
	            self.forward(x_s[t], h_s[t - 1], C_s[t - 1]) # Forward pass
	            
	        loss += -np.log(y_s[t][targets[t], 0]) # Loss for at t
	        
	    self.clear_gradients()

	    dh_next = np.zeros_like(h_s[0]) #dh from the next character
	    dC_next = np.zeros_like(C_s[0]) #dh from the next character

	    for t in reversed(range(len(inputs))):
	        # Backward pass
	        dh_next, dC_next = self.backward(target = targets[t], dh_next = dh_next,
	                     dC_next = dC_next, C_prev = C_s[t-1],
	                     z = z_s[t], f = f_s[t], i = i_s[t], C_bar = C_bar_s[t],
	                     C = C_s[t], o = o_s[t], h = h_s[t], v = v_s[t],
	                     y = y_s[t])

	    self.clip_gradients()
	        
	    return loss, h_s[len(inputs) - 1], C_s[len(inputs) - 1]


	def sample(self,h_prev, C_prev, first_char_idx, sentence_length):
	    x = np.zeros((X_size, 1))
	    x[first_char_idx] = 1

	    h = h_prev
	    C = C_prev

	    indexes = []
	    
	    for t in range(sentence_length):
	        _, _, _, _, C, _, h, _, p = self.forward(x, h, C)
	        idx = np.random.choice(range(X_size), p=p.ravel())
	        x = np.zeros((X_size, 1))
	        x[idx] = 1
	        indexes.append(idx)

	    return indexes

	def update_paramters(self):
		params = self.parameters
		for p in params.all():
			p.m += p.d * p.d # Calculate sum of gradients
	        p.v += -(learning_rate * p.d / np.sqrt(p.m + 1e-8))



#main function to train the model
def train(data,num_epochs):
	#initialise parameters
	parameters = Parameters()
	#initilise lstm module
	lstm = LSTM(parameters)
	#exponential avg of loss
	smooth_loss = -np.log(1.0 / X_size) * T_steps
	num_steps = (len(data)-1)/T_steps
	epoch_losses = []
	for epoch in range(num_epochs):
		p = 0 #	 go from start of data
		losses = []
		g_h_prev = np.zeros((H_size, 1))
		g_C_prev = np.zeros((H_size, 1))

		for step in range(num_steps):
			inputs = [char_to_idx[ch] for ch in data[p:p+T_steps]]
			targets = [char_to_idx[ch] for ch in data[p+1:p+T_steps+1]]
			loss, g_h_prev, g_C_prev = lstm.forward_backward(inputs, targets, g_h_prev, g_C_prev)
			smooth_loss = smooth_loss * 0.999 + loss * 0.001
			losses.append(smooth_loss)
			lstm.update_paramters()
			p += T_steps 

		if (epoch % 1 == 0 and step==num_steps-1):
			inputs = [char_to_idx[ch] for ch in data[p:p+T_steps]]
			sample_idx = lstm.sample(g_h_prev, g_C_prev, inputs[0], 200)
			txt = ''.join(idx_to_char[idx] for idx in sample_idx)
			print("----\n %s \n----" % (txt, ))
			print("epoch %d, loss %f" % (epoch, smooth_loss))

		epoch_losses.append(np.mean(losses))
	return epoch_losses
	 


if __name__ == "__main__":
	losses = train(data,num_epochs=50)
	plt.plot(losses)
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.savefig("lstm.png")
