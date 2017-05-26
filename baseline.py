
import python_speech_features
import tensorflow as tf 
import numpy as np
import scipy.io.wavfile as wav
import os
from os import listdir
from os.path import isfile, join, exists
import wave
from scikits.audiolab import Sndfile
from scikits import audiolab
from unpickle import unpickle
from feature_extractor import *



BATCHES = 1

LR = 0.01
NUM_EPOCHS = 3000
REG = 0.00001
VAL_SPLIT = 0.15


#RNN Params
RNN_Units = 3
RUN_ON_FINAL_RNN_STATE = True


class Baseline():
	def __init__(self):
		pass

	def setup_input(self, max_time, features):
		self.X_placeholder = tf.placeholder(tf.float32, (None, max_time, features))
		self.y_placeholder = tf.placeholder(tf.float32, (None, ))
		self.seq_lens_placeholder = tf.placeholder(tf.float32, (None, ))


	def setup_cnn_graph(self):
		pass

	def setup_hybrid_graph(self):
		pass

	def setup_rnn_graph(self):
		rnn_cell = tf.contrib.rnn.BasicRNNCell(RNN_Units, activation = tf.nn.relu) #COULD TRY OTHER CELLS

		rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, self.X_placeholder, self.seq_lens_placeholder, dtype=tf.float32)

		if(RUN_ON_FINAL_RNN_STATE):
			final_inputs = final_state
		else:
			final_inputs = tf.contrib.layers.flatten(rnn_outputs)

		self.outputs = tf.squeeze(tf.contrib.layers.fully_connected(final_inputs, num_outputs = 1, activation_fn = None, biases_initializer = tf.zeros_initializer()))
		
		degenerate_score = tf.reduce_sum(tf.losses.mean_squared_error(self.y_placeholder, 0.5*tf.ones_like(self.y_placeholder)))
		self.unreg_losses = tf.losses.mean_squared_error(self.y_placeholder, self.outputs) #PLAY AROUND WITH
		self.unreg_loss = tf.reduce_sum(self.unreg_losses)
		self.advantage = degenerate_score - self.unreg_loss
		

	def setup_loss_and_train(self):
		self.l2_loss = REG*tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
		self.loss = self.unreg_loss #+ self.l2_loss
		global_step = tf.Variable(0, trainable=False)
		decaying_lr = tf.train.exponential_decay(LR, global_step, 1000000, 0.95, staircase=True)
		optimizer = tf.train.AdamOptimizer(decaying_lr)
		#optimizer = tf.train.GradientDescentOptimizer(LR)
		self.train_step = optimizer.minimize(self.loss + self.l2_loss, global_step=global_step)


	def get_batch(self, X, y, seq_lens, batch=True, batches = BATCHES):
		total_examples = len(seq_lens)	
		if not batch:
			yield (X, y, seq_lens)
		batch_len = total_examples//batches
		divide = 0
		while divide < total_examples:
			next_divide = min([divide+batch_len, total_examples])
			yield (X[divide:next_divide], y[divide:next_divide], seq_lens[divide:next_divide])
			divide = next_divide

	def prepare(self, features_dir, labels_dir):
		feature_files = get_files_in_dir(features_dir)
		#features = get_files_in_dir(features_dir)
		num_fns = None
		X_unpadded, y = [], []
		for feature_fn in feature_files:
			try:
				label_fn = feature_fn.split('.')[0]
				with open(join(labels_dir, label_fn + '.txt')) as lfn:
					y.append(get_label(lfn.readline()))

				with open(join(features_dir, feature_fn)) as ffn:
					data = np.loadtxt(ffn, delimiter=',')
					print data.shape
					num_fns = data.shape[1]
					X_unpadded.append(data.flatten())
			except IOError as e:
				print "ERROR", e
				continue

		N = len(y)
		y = np.array(y)
		seq_lens = map(len, X_unpadded)
		max_X = max(seq_lens)
		X = np.zeros((N, max_X))

		for n in range(N):
			x_len = seq_lens[n]
			X[n, 0:x_len] += np.array(X_unpadded[n])
		X = X.reshape((N, -1, num_fns))
		assert(X.shape[1] == max_X/num_fns)

		self.setup_input(max_X/num_fns, num_fns)
		# self.setup_cnn_graph(max_X/num_fns, num_fns)
		self.setup_rnn_graph()
		self.setup_loss_and_train()
		return X, y, seq_lens

	def split_data(self, X, y, seq_lens):
		seq_lens = np.array(seq_lens)
		N = len(seq_lens)
		N_val = int(VAL_SPLIT*N)
		val_r = np.random.choice(range(N), (N_val,))
		train_r = set(range(N)) - set(val_r)
		val_r = np.array(val_r)
		train_r = np.array(list(train_r), dtype = np.int)
		X_tr = X[train_r]
		X_val = X[val_r]
		y_tr = y[train_r]
		y_val = y[val_r]
		sq_tr = seq_lens[train_r]
		sq_val = seq_lens[val_r]
		return X_tr, X_val, y_tr, y_val, sq_tr, sq_val

	def get_accuracy(self, Y, pred):
		correct = []
		for i, y in enumerate(Y):
			pred_i = round(max([1, min([0, pred[i]])]))
			if pred_i == y:
				correct.append(1)
			else:
				correct.append(0)
		return sum(correct)*1.0/len(correct)

	def run_baseline(self, features_dir, labels_dir):
		X, y, seq_lens = self.prepare(features_dir, labels_dir)
		X_train, X_val, y_train, y_val, seq_lens_train, seq_lens_val = self.split_data(X, y, seq_lens)

		with tf.Session() as session:
			session.run(tf.global_variables_initializer())
			for epoch in range(NUM_EPOCHS):
				for batch in self.get_batch(X_train, y_train, seq_lens_train):
					X_batch, y_batch, seq_lens_batch = batch
					train_feed_dict = {self.X_placeholder: X_batch, self.y_placeholder : y_batch, self.seq_lens_placeholder : seq_lens_batch}

					_, l2_loss, unreg_loss = session.run([self.train_step, self.l2_loss, self.unreg_loss], train_feed_dict)
					total_loss = l2_loss + unreg_loss

					val_feed_dict = {self.X_placeholder: X_val, self.seq_lens_placeholder: seq_lens_val, self.y_placeholder : y_val}
					adv, val_loss, output = session.run([self.advantage, self.loss, (self.outputs)], val_feed_dict)

					print "accuracy: :", self.get_accuracy(y_val, output)

					#Advantage is the performance above a degenerate algorithm (predict one class every time)
					#print epoch, 'val_loss', val_loss, '(unreg:', val_unreg_loss, ')', ' -- train_loss: ', total_loss, '(', "unreg:", unreg_loss, ')'
					print adv ,unreg_loss, zip(output[0:5], y_val[0:5])

if __name__ == '__main__':
	baseline = Baseline()
	baseline.run_baseline(MFCC_DIR, LABELS_DIR)

