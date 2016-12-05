# Graph-based POS-tagging for low resource languages
# Part 1: Graph Construction

# Alexandra Arkut, Janosch Haber, Muriel Hol, Victor Milewski
# v.0.01 - 2016-11-23

from itertools import chain
import numpy as np
import sys
import time
import pygtrie as trie

class Graphs:
    max_lines = 200000
    english_wordlist = []
    foreign_wordlist = []
    foreign_penta_trie = None
    # english_embeddings = None
    # foreign_embeddings = None
    # pentagrams = []
    # vectors = None
    w = None
    a = None
    r = None

    def __init__(self,en_filename,for_filename):
        print "create english wordlist..."
        start = time.clock()
        self.english_wordlist = self.create_wordlist(en_filename)
        print "finsished in "+str(time.clock()-start)+"s"
        print "length english dict: "+ str(len(self.english_wordlist))
        print "create foreign wordlist..."
        start = time.clock()
        self.foreign_wordlist = self.create_wordlist(for_filename)
        print "finsished in "+str(time.clock()-start)+"s"
        print "length foreing dict: "+ str(len(self.foreign_wordlist))
        print "create trie for foreign pentagrams..."
        start = time.clock()
        self.foreign_penta_trie = self.create_penta_tries(for_filename)
        print "finsished in "+str(time.clock()-start)+"s"

        # print "create english embeddings..."
        # start = time.clock()
        # self.english_embeddings = self.create_embeddings(embedding_dims,self.english_wordlist)
        # print "finsished in "+str(time.clock()-start)+"s"
        # print "create foreign embeddings..."
        # start = time.clock()
        # self.foreign_embeddings = self.create_embeddings(embedding_dims,self.foreign_wordlist)
        # print "finsished in "+str(time.clock()-start)+"s"

        # #nog niet getest vanaf hier
        # print "create pentagrams..."
        # start = time.clock()
        # pentagrams = self.create_pentagrams(for_filename)
        # print "finsished in "+str(time.clock()-start)+"s"
        # print "create feature vectors..."
        # start = time.clock()
        # vectors = create_pentagram_vectors(foreign_embeddings,foreign_wordlist,pentagrams)
        # print "finsished in "+str(time.clock()-start)+"s"
        # print "create weight matrix..."
        # start = time.clock()
        # w = create_weights(vectors)
        # print "finsished in "+str(time.clock()-start)+"s"
        # print "create allignment matrix..."
        # start = time.clock()
        # a = create_allignments(en_filename,for_filename,english_wordlist,foreign_wordlist)
        # print "finsished in "+str(time.clock()-start)+"s"

    # def __init__(self,en_wordlist_file,for_wordlist_file,en_emb_file,for_emb_file,penta_file,vectors_file):
    #     """
    #     read in all the pretrained/prebuild lists,featueres.
    #     store these again in there variables and build the matrices with them
    #     """


    # def __init__(self,en_wordlist_file,for_wordlist_file,en_emb_file,for_emb_file,penta_file,vectors_file,w_file,a_file,r_file):
    #     """
    #     read in all the pretrained/prebuild lists,featueres and matices.
    #     store these again in there variables
    #     """

    def create_penta_tries(self,filename):
        tree = trie.StringTrie()

        with open(filename) as f:
            i = 0
            for line in f:
                if i >= self.max_lines:
                    break
                words = line.split()
                grams = self.context2(2,words)
                for gram in grams:
                    key = ""
                    for g in gram:
                        key += "/"+g
                        try:
                            tree[key] += 1
                        except KeyError:
                            tree[key] = 1
                        # if tree.has_key(key):
                        #     tree[key] += 1
                        # else:
                        #     tree[key] = 1
                i += 1
        return tree

    def create_wordlist(self,filename):
        words = set()
        with open(filename) as f:
            for i,line in enumerate(f):
                if i >= self.max_lines:
                    break
                line = line.lower()
                w = line.split()
                words |= set(w)
        return list(words)

    def create_embeddings(self,embedding_dims,wordlist):
        emb = np.random.normal(0., 0.01, embedding_dims*len(wordlist))
        print emb
        return emb.reshape(len(wordlist),embedding_dims)

    def create_pentagrams(self,filename):
        grams = set()
        with open(filename) as f:
            for line in f:
                grams |= set(self.context2(2,line))
        return grams

    def create_pentagram_vectors(self,embeddings,wordlist,pentagrams):
        vectors = []
        for gram in pentagrams:
            emb = []
            for word in gram:
                vector.append(embeddings[wordlist.index(word)])
            vector = emb
            vector += get_trigram(emb)
            vector += get_context(emb,'l')
            vector += get_context(emb,'r')
            vector += get_center_word(emb)
            vector += get_context_nocenter(emb)
            vector += word_context(emb,'l')
            vector += word_context(emb,'r')
            vector += has_suffix(emb)
            vectors.append(vector)
        return np.array(vectors)


    def create_weights(self,vectors):
        w = np.zeros(len(vector)**2).reshape(len(vectors),len(vectors))
        for i,vecA in enumerate(vectors):
            for j,vecB in enumerate(vectors):
                w[i,j] = cosine_similarity(vecA,vecB)
        return w

    # def create_allignments(self,en_filename,for_filename,english,foreign):
    #     a = np.zeros(len(en_filename)*len(for_filename).reshape(len(for_filename),len(en_filename))
    #     with open(en_filename) as fen and open(for_filename) as ffor:
    #         for en_line in fen and for_line in ffor:
    #             align = run_allignment_tool(en_line,for_line)
    #             for i,j in enumerate(align):
    #                 a[i,j] = a[i,j]+1
    #     return a


    # context with <s> tags added. n is the ammount of context, left and right
    def context2(self,n, words):
        narray = [None]*(len(words))
        for i in range(len(words)):
            gram = []
            if i-n < 0:
                for _ in range(0-(i-n)):
                    gram += ['<S>']
            gram += words[i-n+len(gram):i]
            if i+n > len(words)-1:
                gram += words[i:len(words)]
                for _ in range(2*n+1-len(gram)):
                    gram += ['</s>']
            else:
                gram += words[i:i+n+1]
            narray[i] = gram
        return(narray)

    # get left and right context words from pentagram
    def get_context(self,pentagram,lr):
        if lr == 'l':
            return(pentagram[0:2])
        if lr == 'r':
            return(pentagram[len(pentagram)-2:len(pentagram)])

    def get_trigram(self,pentagram):
        return([pentagram[1],pentagram[2],pentagram[3]])

    # get token word
    def get_center_word(self,pentagram):
        return(pentagram[2])

    # get trigram without token
    def get_context_nocenter(self,pentagram):
        return([pentagram[1],pentagram[3]])

    # get left/right word and right/left context
    def word_context(self,pentagram,lr):
        if lr =='l':
            return([pentagram[1],pentagram[3],pentagram[4]])
        if lr == 'r':
            return([pentagram[0],pentagram[1],pentagram[3]])

    # something with suffix
    #how do we know what the suffixes are for the language
    #isn't the suffix count not always zero or equal to center word
    def has_suffix(self,pentagram):
        return 0

    def cosine_similarity(vecA,vecB):
        return np.dot(vecA,vecB)/(np.dot(vecA,vecA)*np.dot(vecB,vecB))

    def get_weight_matrix(self):
        return self.w

    def get_allignment_matrix(self):
        return self.a

    def get_r_matrix(self):
        return self.r

    def get_foreign_embeddings(self):
        return self.foreign_embeddings

    def get_english_embeddings(self):
        return self.english_embeddings

    def get_foreign_wordlist(self):
        return self.foreign_wordist

    def get_english_wordlist(self):
        return self.english_wordist


    # # not used
    # def ngram(n, words):
    #     narray = [None]*(len(words)+1-n)
    #     for i in range(len(words)+1-n):
    #         narray[i] = [words[i:i+n]]
    #     return(narray)
    
    # # context with <s> tags added
    # def context(n, words):
    #     narray = [None]*(len(words))
    #     for i in range(len(words)):
    #         narray[i] = [words[i-n:i+n+1]]
    #         if i-n < 0:
    #             narray[i] = ['<S>']+words[i:i+n+1]
    #         elif i+n > len(words)-1:
    #             narray[i] = words[i-n:len(words)]+['</S>']
    #             return(narray)
    #     return(narray)

    # def create_gram_features():
    #     #create all the feature vectors from the counts
    #     grams = []
    #     for gram,penta_count in pentagram_count.most_common():
    #         feature_vec = []
    #         feature_vec.append(penta_count)
    #         feature_vec.append(trigram_count[str(get_trigram(gram))])
    #         feature_vec.append(left_context_count[str(get_context(gram,'l'))])
    #         feature_vec.append(right_context_count[str(get_context(gram,'r'))])
    #         feature_vec.append(center_count[str(get_center_word(gram))])
    #         feature_vec.append(trigram_nocenter_count[str(get_context_nocenter(gram))])
    #         feature_vec.append(lw_rc_count[str(word_context(gram,'l'))])
    #         feature_vec.append(rw_lc_count[str(word_context(gram,'r'))])
    #         feature_vec.append(has_suffix(gram))
    #         grams.append((gram,feature_vec))
    #     return grams


    # def get_weight_matrix(filename):
    #     gram_features = create_gram_features()
    #     print len(gram_features)
    #     return create_weight_matrix(gram_features)

