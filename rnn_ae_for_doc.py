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
		self.label_arr = None

	def load_docs(self, file_name):
		self.data_list = []
		self.data_len_list = []
		label_list = []

		with open(file_name, 'r') as f:
			for line in f:
				label, context = tuple(line.split('\t'))
				label_list += [int(label)]
				self.data_list += [context.rstrip().split(' ')]
				self.data_len_list += [len(self.data_list[-1]) + 2] # The 2 is added for '<sos>' and '<eos>'.

		self.label_arr = np.array(label_list)

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

			self.orig_full_state = tf.concat([tf.concat(fw_state_tuple, axis=1), tf.concat(bw_state_tuple, axis=1)], axis=1)

			temp_full_state = self.orig_full_state
			for layer_idx in range(mid_dense_layers):
				temp_full_state = tf.layers.dense(temp_full_state, 2 * rnn_hidden_dims)

			self.full_state = temp_full_state

		with tf.name_scope('decoder'):

			fw_init_h = tf.slice(self.full_state, [0, 0], [-1, rnn_hidden_dims])
			fw_init_c = tf.slice(self.full_state, [0, rnn_hidden_dims], [-1, rnn_hidden_dims])
			fw_state = tf.nn.rnn_cell.LSTMStateTuple(fw_init_c, fw_init_h)

			fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_dims)

			projection_layer = tf.layers.Dense(units = token_num)

			helper = tf.contrib.seq2seq.TrainingHelper(doc_embed, self.input_doc_len)
			decoder = tf.contrib.seq2seq.BasicDecoder(fw_lstm_cell, helper, fw_state, output_layer = projection_layer)
			output, _, __ = tf.contrib.seq2seq.dynamic_decode(decoder)

		with tf.name_scope('get_loss'):

			logits = output.rnn_output

			self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_doc, logits=logits))

	def train_model(self, learning_rate, epochs, batch_size, data_arr, data_len_arr, test_data_arr, test_data_len_arr, test_data2_arr, test_data2_len_arr, label_arr, test_label_arr, test_label2_arr, max_gradient, epoch_per_save, epoch_per_eval, model_file_name):

		self.batch_size = batch_size
		self.data_num = len(data_arr)
		self.max_gradient = max_gradient
		
		feed_dict = {self.input_doc: data_arr, self.input_doc_len: data_len_arr}
		test_feed_dict = {self.input_doc: test_data_arr, self.input_doc_len: test_data_len_arr}
		test2_feed_dict = {self.input_doc: test_data2_arr, self.input_doc_len: test_data2_len_arr}

		# These are to be used in evaluation.
		self.existing_doc_feed_dict = self.concat_dict(feed_dict, test_feed_dict)
		self.new_doc_feed_dict = test2_feed_dict
		self.existing_doc_label_arr = np.concatenate([label_arr, test_label_arr], axis = 0)
		self.new_doc_label_arr = test_label2_arr

		batch_num = self.data_num // batch_size

		with tf.Session(config = config) as self.sess:

			optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
			operation = self.clip_gradients(optimizer)
			self.sess.run(tf.global_variables_initializer())

			for epoch_idx in range(epochs):
				print ('training epoch ', epoch_idx)

				feed_dict = self.shuffle_dict(feed_dict)

				for batch_idx in range(batch_num):
					# print ('batch ', batch_idx)
					history = self.sess.run(operation, self.get_batch(feed_dict, batch_idx * batch_size))

				if epoch_idx % epoch_per_eval == 0:

					print ('evaluating...')
					print ('training loss: ', self.run_value_by_batch_size(self.loss, feed_dict))
					print ('testing  loss: ', self.run_value_by_batch_size(self.loss, test_feed_dict))
					self.evaluate(use_orig_full_state = 0, doc_slice_num = 10)

				if epoch_idx % epoch_per_save == 0:
					saver = tf.train.Saver()
					saver.save(self.sess, model_file_name)

					print ('model saved as ', model_file_name)

	def evaluate(self, use_orig_full_state, doc_slice_num):

		embed_placeholder = self.orig_full_state if use_orig_full_state else self.full_state

		existing_doc_embed, _ = self.run_matrix_by_batch_size(embed_placeholder, feed_dict = self.existing_doc_feed_dict)
		new_doc_embed, true_len = self.run_matrix_by_batch_size(embed_placeholder, feed_dict = self.new_doc_feed_dict)

		existing_doc_embed_norm = np.expand_dims(np.linalg.norm(existing_doc_embed, axis = 1), axis = 1)
		new_doc_embed_norm = np.expand_dims(np.linalg.norm(new_doc_embed, axis = 1), axis = 1)

		dot = np.matmul(new_doc_embed, np.transpose(existing_doc_embed))
		cosine_sim = dot / new_doc_embed_norm / np.transpose(existing_doc_embed_norm)

		get_label = lambda x: self.existing_doc_label_arr[x]
		vfunc = np.vectorize(get_label)

		correct = np.equal(vfunc(np.argsort(-cosine_sim,axis=1)), np.expand_dims(self.new_doc_label_arr[:true_len],axis=1))

		slice_size = len(self.existing_doc_label_arr) // doc_slice_num
		
		proportion_list = [(slice_idx + 1) / doc_slice_num for slice_idx in range(doc_slice_num)]
		accuracy_list = [np.mean(correct[:, :(slice_idx + 1) * slice_size]) for slice_idx in range(doc_slice_num)]

		print ('proportion\taccuracy')
		for pair in zip(proportion_list, accuracy_list):
			print (pair)

	def run_matrix_by_batch_size(self, variable, feed_dict):

		batch_num = len(feed_dict[list(feed_dict.keys())[0]]) // self.batch_size
		matrix_list = []
		
		for batch_idx in range(batch_num):
			matrix_list += [self.sess.run(variable, feed_dict = self.get_batch(feed_dict, batch_idx * self.batch_size))]

		return np.concatenate(matrix_list, axis = 0), batch_num * self.batch_size

	def run_value_by_batch_size(self, variable, feed_dict):

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

	def concat_dict(self, dict1, dict2):

		key_list = list(dict1.keys())

		return {key: np.concatenate([dict1[key], dict2[key]], axis = 0) for key in key_list}

if __name__ == '__main__':

	global_doc_max_len = 100

	print ('loading training data...')
	data_formater = Data_formater()
	data_formater.load_docs('./tm/train.txt')
	data_formater.build_dict()
	data_formater.data2format(doc_max_len = global_doc_max_len)
	data_formater.list2arr()

	print ('loading testing data...') # testing data here is actually validating data.
	test_data_formater = Data_formater()
	test_data_formater.load_docs('./tm/held_out.txt')
	test_data_formater.load_dict(data_formater.word2idx, data_formater.token_num)
	test_data_formater.data2format(doc_max_len = global_doc_max_len)
	test_data_formater.list2arr()

	print ('loading testing data2...') # testing data2 is the real testing data.
	test_data2_formater = Data_formater()
	test_data2_formater.load_docs('./tm/test.txt')
	test_data2_formater.load_dict(data_formater.word2idx, data_formater.token_num)
	test_data2_formater.data2format(doc_max_len = global_doc_max_len)
	test_data2_formater.list2arr()

	print ('building model....')
	ae_model = AE_model(w_stddev = 0.1, b_stddev = 0.01)

	ae_model.build_model(doc_max_len = data_formater.doc_max_len, token_num = data_formater.token_num, embed_dims = 16, \
			rnn_hidden_dims = 32, mid_dense_layers = 2)

	ae_model.train_model(learning_rate = 0.01, epochs = 1000, batch_size = 32, \
			data_arr = data_formater.data_arr, data_len_arr = data_formater.data_len_arr, \
			test_data_arr = test_data_formater.data_arr, test_data_len_arr = test_data_formater.data_len_arr, \
			test_data2_arr = test_data2_formater.data_arr, test_data2_len_arr = test_data2_formater.data_len_arr, \
			label_arr = data_formater.label_arr, test_label_arr = test_data_formater.label_arr, \
			test_label2_arr = test_data2_formater.label_arr, \
			max_gradient = 3.0, epoch_per_save = 20, epoch_per_eval = 10, model_file_name = './model_test.ckpt', )