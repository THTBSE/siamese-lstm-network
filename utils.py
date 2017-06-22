#-*- coding:utf-8 -*-
"""
应用于Siamese LSTM的data util
输入文本为清洗好的文本,格式为
seq1_token1 seq1_token2 seq1_token2 ... seq1_tokenN\tseq2_token1 seq2_token2 seq2_token3 ... seq2_tokenN\tlabel
文本1与文本2以及label用"\t"隔开
文本之间的token使用空格" "隔开
label为0或1表示相似与不相似
"""

import os
import cPickle
import numpy as np
from collections import defaultdict

class InputHelper():

	def __init__(self, data_dir, input_file, batch_size, sequence_length, is_train=True):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.sequence_length = sequence_length

		vocab_file = os.path.join(data_dir, 'vocab.pkl')
		input_file = os.path.join(data_dir, input_file)

		if not (os.path.exists(vocab_file)):
			print 'readling train file'
			self.preprocess(input_file, vocab_file)
		else:
			print 'loading vocab file'
			self.load_vocab(vocab_file)

		if is_train:
			self.create_batches(input_file)
			self.reset_batch()

	def preprocess(self, input_file, vocab_file, min_freq=2):

		token_freq = defaultdict(int)

		for line in open(input_file):
			seq1,seq2,label = line.rstrip().split('\t')
			seq = seq1 + ' ' + seq2
			for token in seq.split(' '):
				token_freq[token] += 1

		token_list = [w for w in token_freq.keys() if token_freq[w] >= min_freq]
		token_list.append('<pad>')
		token_dict = {token:index for index,token in enumerate(token_list)}

		with open(vocab_file, 'w') as f:
			cPickle.dump(token_dict, f)

		self.token_dictionary = token_dict
		self.vocab_size = len(self.token_dictionary)

	def load_vocab(self, vocab_file):

		with open(vocab_file, 'rb') as f:
			self.token_dictionary = cPickle.load(f)
			self.vocab_size = len(self.token_dictionary)

	def text_to_array(self, text, is_clip=True):

		seq_ids = [int(self.token_dictionary.get(token)) for token in text if self.token_dictionary.get(token) is not None]
		if is_clip:
			seq_ids = seq_ids[:self.sequence_length]
		return seq_ids

	def padding_seq(self, seq_array, padding_index):

		for i in xrange(len(seq_array), self.sequence_length):
			seq_array.append(padding_index)

	def create_batches(self, text_file):

		x1 = []
		x2 = []
		y = []

		padding_index = self.vocab_size - 1
		for line in open(text_file):
			seq1, seq2, label = line.rstrip().split('\t')
			seq1_array = self.text_to_array(seq1.split(' '))
			seq2_array = self.text_to_array(seq2.split(' '))

			self.padding_seq(seq1_array, padding_index)
			self.padding_seq(seq2_array, padding_index)

			label = int(label)
			x1.append(seq1_array)
			x2.append(seq2_array)
			y.append(label)

		x1 = np.array(x1)
		x2 = np.array(x2)
		y = np.array(y)

		self.num_samples = len(y)
		self.num_batches = self.num_samples / self.batch_size
		indices = np.random.permutation(self.num_samples)
		self.x1 = x1[indices]
		self.x2 = x2[indices]
		self.y = y[indices]

	def next_batch(self):
		
		begin = self.pointer
		end = self.pointer + self.batch_size
		x1_batch = self.x1[begin:end]
		x2_batch = self.x2[begin:end]
		y_batch = self.y[begin:end]

		new_pointer = self.pointer + self.batch_size

		if new_pointer >= self.num_samples:
			self.eos = True
		else:
			self.pointer = new_pointer

		return x1_batch, x2_batch, y_batch

	def reset_batch(self):
		self.pointer = 0
		self.eos = False

if __name__ == '__main__':

	data_loader = InputHelper('data','train',128, 20)
	x1,x2,y = data_loader.next_batch()
	print x1[0]
	print type(x1[0])