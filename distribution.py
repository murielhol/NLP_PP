
from collections import Counter
import csv
import io
import numpy as np 
import pickle


# def collect(filename):
#     """
#     create a list of all the unique tokens in the given file
#     return the list and total number of tokens
#     """
#     words = []
#     tags = []
#     with open(filename) as f:
#         for i,line in enumerate(f):
#             line = line.lower()
#             w = line.split()
#             word = w[0]
#             tag = w[1]

	 
#     return words, tags
def collect(file):
	words = []
	tags = []
	with open(file, "rU", encoding='utf-8') as digit_file:
		reader = csv.reader(digit_file, delimiter = ",")
		for line in reader:
			try:
				w = line[0].encode('ascii', 'ignore').decode('ascii')
				t = line[1]
				words.append(w)
				tags.append(t)
			except:
				print(line)
				continue
	return words, tags


# def onehot_tags(words, tags, tag_set):
# 	N = len(words)
# 	zero_hot = np.zeros((len(words), len(tag_set)))
# 	for i in range(N):
# 		index = tag_set.index(tags[i])
# 		zero_hot[i][index] = 1

# 	return zero_hot

# # # get the bigarms
# def bigram(words):
# 	N = len(words)
# 	# attach start sign after each full stop and before the first sentence
# 	words = np.array(words)
# 	words = np.insert(words, 0, '<s>')
# 	#tags = 	np.insert(tags, 0, '<s>')
# 	# check were to insert a start sign
# 	insert_list=[]
# 	for i in range(N):
# 		if(words[i] == '.'):
# 			insert_list.append(i+1)
# 	# insert
# 	j = 0
# 	for i in insert_list:
# 		words = np.insert(words, i+j, '<s>')
# 		#tags = np.insert(tags, i+j, '<s>')
# 		j = j+1
# 	# now collect bigrams
# 	bigram_list = []

# 	N = len(words)
# 	for i in range(N):
# 		if words[i] != '<s>':
# 			bigram = str(words[i] + '|' + words[i-1])
# 			bigram_list.append(bigram)
	
# 	return bigram_list



def distribute(words, tags, tag_set):

	# dict all combination of tags
	tag_dict = {}
	for i in range(int(len(words))):
		tag_dict.setdefault(words[i], []).append(tags[i])
	# assign distribution to each unique word
	word_set = set(words)
	d = []
	n = len(word_set)
	i =1
	for word in word_set:
		print(i/n)
		i = i+1
		tag_list = tag_dict[word]
		bin_list = [tag_list.count(x) for x in tag_set]
		s = sum(bin_list)
	 	#normalize 
		distribution = [x/s for x in bin_list]
		distribution.insert(0, word)
		d.append(distribution)
	tag_set.insert(0,'words')
	d.insert(0,tag_set)
	return d, word_set
	

if __name__ == "__main__":

	filename = "/Users/murielhol/NLP/en_50000_unitagged_good.txt"

	# collect the words and tags
	words, tags = collect(filename)
	tag_set = list(set(tags))

	# print(words)
	# print(tags)
	# print(tag_set)

	# # convert into bigrams
	# # bigrams = bigram(words)
	# # print("bigrams are made")
	# # get the tag distributions of each word, based on its bigram

	d, word_set = distribute(words, tags, tag_set)
	outfile = open('d_unitag_matrix_en50000.txt','w', encoding='utf-8')
	writer=csv.writer(outfile)
	for i in range(len(word_set)):
		writer.writerow(d[i])


	pickle.dump( d, open( "d_unitag_matrix_en50000.p", "wb" ) )


	# ###########################################################################
	# # un-commend if you want to create a pickle/txt matrix with 
	# # one-hot vector encoding of the POS tags

	# onehot_matrix = onehot_tags(words, tags, tag_set)

	# outfile = open('onehot_unitag_matrix_en50000.txt','w', encoding='utf-8')
	# writer=csv.writer(outfile)

	# for i in range(len(tags)-1):
	# 	writer.writerow(onehot_matrix[i])


	# pickle.dump( onehot_matrix, open( "onehot_unitag_matrix_en50000.p", "wb" ) )







