# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-04-27 15:18:00
# @Last Modified by:   vamshi
# @Last Modified time: 2018-04-29 12:00:13

import numpy as np
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
import sys
from numpy import linalg as LA
import matplotlib.pyplot as plt

class RNN:

	def __init__(self,vocab_size,hidden_size,seq_length):
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.seq_length = seq_length

		self.Wxh = np.random.uniform(-np.sqrt(1.0/vocab_size), np.sqrt(1.0/vocab_size), size=(hidden_size,vocab_size))
		self.Whh = np.random.uniform(-np.sqrt(1.0/hidden_size), np.sqrt(1.0/hidden_size), size=(hidden_size,hidden_size))
		self.Why = np.random.uniform(-np.sqrt(1.0/hidden_size), np.sqrt(1.0/hidden_size), size=(vocab_size,hidden_size))
		self.bh = np.zeros((hidden_size, 1)) # hidden bias
		self.by = np.zeros((vocab_size, 1)) # output bias


	def forward(self,inputs,targets,hprev,temperature=0.5):
		"""
		inputs,targets are both list of integers.
		hprev is Hx1 array of initial hidden state
		returns the loss, gradients on model parameters, and last hidden state
		"""
		xs, hs, ys, ps = {}, {}, {}, {}
		hs[-1] = np.copy(hprev)
		loss = 0
		# forward pass
		for t in xrange(len(inputs)):
			xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
			xs[t][inputs[t]] = 1
			hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
			ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
			ps[t] = np.exp(ys[t]/temperature) / np.sum(np.exp(ys[t]/temperature)) # probabilities for next chars
			loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

		return xs,ps,hs,loss


	def bptt(self,inputs,targets,hprev,temperature=1.0):

		xs,ps,hs,loss = self.forward(inputs, targets, hprev)
		# backward pass: compute gradients going backwards
		dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
		dhnext = np.zeros_like(hs[0])
		for t in reversed(xrange(len(inputs))):
			dy = np.copy(ps[t])
			dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
			dWhy += np.dot(dy, hs[t].T)
			dby += dy
			dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
			dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
			dbh += dhraw
			dWxh += np.dot(dhraw, xs[t].T)
			dWhh += np.dot(dhraw, hs[t-1].T)
			dhnext = np.dot(self.Whh.T, dhraw)
		for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]


	def sample(self,h, seed_char, n):
		""" 
		sample a sequence of integers from the model 
		h is memory state, seed_ix is seed letter for first time step
		"""
		x = np.zeros((vocab_size, 1))
		x[seed_char] = 1
		ixes = []
		for t in xrange(n):
			h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
			y = np.dot(self.Why, h) + self.by
			p = np.exp(y) / np.sum(np.exp(y))
			ix = np.random.choice(range(vocab_size), p=p.ravel())
			x = np.zeros((vocab_size, 1))
			x[ix] = 1
			ixes.append(ix)
		return ixes



def train(model,data,num_epochs,learning_rate,temperature=1.0,epochs_sample=1,step_disp_progress=1000):
	n, p = 0, 0
	mWxh, mWhh, mWhy = np.zeros_like(model.Wxh), np.zeros_like(model.Whh), np.zeros_like(model.Why)
	mbh, mby = np.zeros_like(model.bh), np.zeros_like(model.by) # memory variables for Adagrad
	smooth_loss = -np.log(1.0/model.vocab_size)*model.seq_length # loss at iteration 0
	num_steps = (len(data)-1)/model.seq_length
	epoch_losses = []

	for epoch in range(num_epochs):
		hprev = np.zeros((model.hidden_size,1)) # reset RNN memory
		p = 0 # go from start of data
		losses = []

		# sample from the model now and then
		if epoch % epochs_sample == 0 and epoch!=0:
			sample_ix = model.sample(hprev, 0 , 100)
			txt = ''.join(ix_to_char[ix] for ix in sample_ix)
			print( 'sampled text:\n----\n %s \n----' % (txt, ))
			
			#test with giving random input
			hpre = np.zeros((model.hidden_size,1)) # reset RNN memory
			po = np.random.randint(0, len(data)-200)
			test_ip = [char_to_ix[ch] for ch in data[po:po+100]]
			test_tar = [char_to_ix[ch] for ch in data[po+1:po+100+1]]

			_,pr,_,_  = model.forward(test_ip, test_tar, hpre)
			test_ix = [np.argmax(x) for x in pr.values()]
			test_txt = ''.join(ix_to_char[ix] for ix in test_ix)
			print("test output:\n----\n %s \n ----"%(test_txt))
			
			with open("train_"+str(model.hidden_size)+"_"+str(model.seq_length)+"_"+str(temperature)+".txt","a") as f:
				f.writelines("##################################\n\n")
				f.writelines("epoch %d\n\n"%(epoch))
				f.writelines("sampled text\n----------------------------\n")
				f.writelines(txt+"\n\n\n")
				f.writelines("tested text\n-----------------------------\n")
				f.writelines("input:\n")
				f.writelines(''.join(ix_to_char[ix] for ix in test_ip))
				f.writelines("\n\noutput:\n")
				f.writelines(test_txt+"\n\n\n")
				f.writelines("###################################\n\n")
				f.close()


		for step in range(num_steps):
			inputs = [char_to_ix[ch] for ch in data[p:p+model.seq_length]]
			targets = [char_to_ix[ch] for ch in data[p+1:p+model.seq_length+1]]


			# forward seq_length characters through the net and fetch gradient
			loss, dWxh, dWhh, dWhy, dbh, dby, hprev = model.bptt(inputs, targets, hprev,temperature)
			smooth_loss = smooth_loss * 0.999 + loss * 0.001
			losses.append(smooth_loss)
			if step % step_disp_progress == 0: print 'epoch %d, iter %d, loss: %f' % (epoch, step, smooth_loss) # print progress

			# perform parameter update with Adagrad
			for param, dparam, mem in zip([model.Wxh, model.Whh, model.Why, model.bh, model.by], 
			                            [dWxh, dWhh, dWhy, dbh, dby], 
			                            [mWxh, mWhh, mWhy, mbh, mby]):
				mem += dparam * dparam
				param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

			p += model.seq_length # move data pointer
		epoch_losses.append(np.mean(losses))
	return epoch_losses


if __name__=='__main__':
	
	# hyperparameters
	hidden_size = 100 # size of hidden layer of neurons
	seq_length = 4 # number of steps to unroll the RNN for
	temperature = 1.0
	learning_rate = 1e-2
	num_epochs = 100

	if(len(sys.argv)!=4):
		print"correct usage temp hidden size seq length"
		sys.exit(1)
	else:
	    temperature = float(sys.argv[1])
	    hidden_size = int(sys.argv[2])
	    seq_length = int(sys.argv[3])

	#get data
	data = open("3stud.txt","r").read()
	chars = list(set(data))
	data_size, vocab_size = len(data), 256
	print 'data has %d characters, %d unique.' % (data_size, len(chars))

	#create dictionaries
	ix_to_char = {i:chr(i) for i in range(256)}
	char_to_ix = {c:ord(c) for c in ix_to_char.values()}

	model = RNN(vocab_size=vocab_size, hidden_size=hidden_size, seq_length=seq_length)
	losses = train(model, data, num_epochs=num_epochs, temperature=temperature, learning_rate=learning_rate,epochs_sample=1,step_disp_progress=1000)

	print("Sampling text after training")
	for i in range(5):
		seed_ix = char_to_ix[data[np.random.randint(0, len(data)-200)]]
		hprev = np.zeros((model.hidden_size,1)) # reset RNN memory
		sample_ix = model.sample(hprev, seed_ix , 100)
		txt = ''.join(ix_to_char[ix] for ix in sample_ix)
		print( 'sampled text:\n----\n %s \n----' % (txt, ))
		with open("train_"+str(model.hidden_size)+"_"+str(model.seq_length)+"_"+str(temperature)+".txt","a") as f:
			f.writelines("\n\nsampling with seed_ix: %d\n----\n %s \n----"%(seed_ix,txt))
			f.close()

	np.save("./losses_"+str(model.hidden_size)+"_"+str(model.seq_length)+"_"+str(temperature), losses)
	plt.plot(losses)
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.title("hidden size: %d,seq length: %d, temperature: %d"%(hidden_size,seq_length,temperature))
	plt.savefig('plt_'+str(model.hidden_size)+"_"+str(model.seq_length)+"_"+str(temperature)+".png")






















