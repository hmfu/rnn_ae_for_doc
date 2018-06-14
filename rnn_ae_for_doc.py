import tensorflow as tf
import numpy as np
import pickle
import os

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

class Data_formater(object):

	def __init__(self):
		self.data_list = None
		self.word2idx = None
		self.doc_len_list = None
		self.doc_max_len = None
		self.data_arr = None
		self.data_len_arr = None
		self.token_num = None

	def load_docs(self, file_name):
		self.data_list = []
		self.data_len_list = []

		with open(file_name, 'r') as f:
			for line in f:
				self.data_list += [line.rstrip().split(' ')]
				self.data_len_list += [len(self.data_list[-1]) + 2] # The 2 is added for '<sos>' and '<eos>'.

	def build_dict(self):
		word_set = set([word for word_list in self.data_list for word in word_list ])
		word_set = word_set.union(set(['<sos>', '<eos>', '<pad>', '<unk>']))
		self.token_num = len(word_set)

		self.word2idx = {}
		idx = 0

		for word in word_set:
			self.word2idx[word] = idx
			idx += 1

	def load_dict(self, word2idx, token_num):
		self.word2idx = word2idx
		self.token_num = token_num

	def data2format(self, doc_max_len):
		self.doc_max_len = doc_max_len
		
		self.data_list = [['<sos>'] + word_list + ['<eos>'] for word_list in self.data_list]
		self.data_list = [word_list[: doc_max_len] if len(word_list) > doc_max_len \
				else word_list + ['<pad>'] * (doc_max_len - len(word_list)) for word_list in self.data_list]
		self.data_list = [[self.word2idx.get(word, self.word2idx['<unk>']) for word in word_list] \
				for word_list in self.data_list]

	def list2arr(self):
		self.data_arr = np.array(self.data_list)
		self.data_len_arr = np.minimum(np.array(self.data_len_list), self.doc_max_len)

class AE_model(object):

	def __init__(self, w_stddev, b_stddev):
		self.w_stddev = w_stddev
		self.b_stddev = b_stddev

	def build_model(self, doc_max_len, token_num, embed_dims, rnn_hidden_dims, mid_dense_layers):

		with tf.name_scope('setup_inputs'):

			self.input_doc = tf.placeholder(dtype = tf.int32, shape = [None, doc_max_len])
			self.input_doc_len = tf.placeholder(dtype = tf.int32, shape = [None])

		with tf.name_scope('idx_to_embedding'):

			embed_table = tf.Variable(tf.random_normal([token_num, embed_dims], dtype = tf.float32, stddev = self.w_stddev))
			doc_embed = tf.nn.embedding_lookup(embed_table, self.input_doc)

		with tf.name_scope('encoder'):

			fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_dims)
			bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_dims)

			output_tuple, state_tuple = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, doc_embed, \
					self.input_doc_len, dtype = tf.float32)

			fw_state_tuple, bw_state_tuple = state_tuple

		with tf.name_scope('mid_dense_layers'):

			full_state = tf.concat([tf.concat(fw_state_tuple, axis = 1), tf.concat(bw_state_tuple, axis = 1)], axis = 1)

			for layer_idx in range(mid_dense_layers):
				full_state = tf.layers.dense(full_state, 2 * rnn_hidden_dims)

		with tf.name_scope('decoder'):

			fw_init_h = tf.slice(full_state, [0, 0], [-1, rnn_hidden_dims])
			fw_init_c = tf.slice(full_state, [0, rnn_hidden_dims], [-1, rnn_hidden_dims])
			fw_state = tf.nn.rnn_cell.LSTMStateTuple(fw_init_c, fw_init_h)

			fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_dims)

			projection_layer = tf.layers.Dense(units = token_num)

			helper = tf.contrib.seq2seq.TrainingHelper(doc_embed, self.input_doc_len)
			decoder = tf.contrib.seq2seq.BasicDecoder(fw_lstm_cell, helper, fw_state, output_layer = projection_layer)
			output, _, __ = tf.contrib.seq2seq.dynamic_decode(decoder)

		with tf.name_scope('get_loss'):

			logits = output.rnn_output

			self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_doc, logits=logits))

	def train_model(self, learning_rate, epochs, batch_size, data_arr, data_len_arr, test_data_arr, test_data_len_arr, max_gradient):

		self.batch_size = batch_size
		self.data_num = len(data_arr)
		self.max_gradient = max_gradient
		
		feed_dict = {self.input_doc: data_arr, self.input_doc_len: data_len_arr}
		test_feed_dict = {self.input_doc: test_data_arr, self.input_doc_len: test_data_len_arr}

		batch_num = self.data_num // batch_size

		with tf.Session(config = config) as self.sess:

			optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
			operation = self.clip_gradients(optimizer)
			self.sess.run(tf.global_variables_initializer())

			for epoch_idx in range(epochs):
				print ('training epoch ', epoch_idx)

				feed_dict = self.shuffle_dict(feed_dict)

				for batch_idx in range(batch_num):
					if batch_idx % 10 == 0:
						print ('training batch ', batch_idx)

					history = self.sess.run(operation, self.get_batch(feed_dict, batch_idx * batch_size))

				print ('evaluating...')
				print ('training loss: ', self.run_by_batch_size(self.loss, feed_dict))
				print ('testing  loss: ', self.run_by_batch_size(self.loss, test_feed_dict))

	def run_by_batch_size(self, variable, feed_dict):

		batch_num = len(feed_dict[list(feed_dict.keys())[0]]) // self.batch_size
		sum_of_loss = 0
		
		for batch_idx in range(batch_num):
			sum_of_loss += self.sess.run(variable, feed_dict = self.get_batch(feed_dict, batch_idx * self.batch_size))

		return sum_of_loss / batch_num

	def clip_gradients(self, optimizer):

		params = tf.trainable_variables()
		gradients = tf.gradients(self.loss, params)
		clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient)

		operation = optimizer.apply_gradients(zip(clipped_gradients, params))

		return operation

	def get_batch(self, dict, start_ind):

		return {k: v[start_ind: start_ind + self.batch_size] for k, v in dict.items()}

	def shuffle_dict(self, dict):

		order = np.random.permutation(self.data_num)

		return {k: v[order] for k, v in dict.items()}

if __name__ == '__main__':

	print ('loading training data...')
	data_formater = Data_formater()
	data_formater.load_docs('./tm/train.txt')
	data_formater.build_dict()
	data_formater.data2format(doc_max_len = 100)
	data_formater.list2arr()

	print ('loading testing data...')
	test_data_formater = Data_formater()
	test_data_formater.load_docs('./tm/test.txt')
	test_data_formater.load_dict(data_formater.word2idx, data_formater.token_num)
	test_data_formater.data2format(doc_max_len = 100)
	test_data_formater.list2arr()

	print ('building model....')
	ae_model = AE_model(w_stddev = 0.1, b_stddev = 0.01)

	ae_model.build_model(doc_max_len = data_formater.doc_max_len, token_num = data_formater.token_num, embed_dims = 16, \
			rnn_hidden_dims = 32, mid_dense_layers = 2)

	ae_model.train_model(learning_rate = 0.01, epochs = 1000, batch_size = 128, data_arr = data_formater.data_arr, \
			data_len_arr = data_formater.data_len_arr, test_data_arr = data_formater.data_arr, \
			test_data_len_arr = data_formater.data_len_arr, max_gradient = 3.0)