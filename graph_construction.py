# Graph-based POS-tagging for low resource languages
# Part 1: Graph Construction

# Alexandra Arkut, Janosch Haber, Muriel Hol, Victor Milewski
# v.0.01 - 2016-11-23

###TODO###
#formulas for calculating different PMIs
#Method for calculating allignment matrix
#method for calculating r matrix

#pmi methods are created, some are solved, some have a proposed solution

from itertools import chain
import numpy as np
from numpy.linalg import norm as n
from scipy.spatial.distance import cosine
#lukt niet op windows... hahaha
#from sklearn.metrics.pairwise import cosine_similarity
import sys
import time
import pygtrie as trie
import pickle
import re

class Graphs:
    #limit data size (remove after development)
    max_lines = 100

    #Total number of tokens in foreign corpus
    total_unigrams = 0

    #List of all the unique english words in the corpus
    english_wordlist = []

    #List of all the unique foreign words in the corpus
    foreign_wordlist = []

    #List of all the unique trigrams in the corpus
    foreign_trigrams = []

    #Trie where all the counts are stored for each n-gram till 5-gram
    foreign_penta_trie = None

    #The vectors that represent the trigrams from foreign_trigrams
    foreign_trigram_vectors = None

    #the weight matrix
    w = None

    #the alignment matrix from english to foreign
    a = None

    #tag distribution for foreign languish from alignment with english
    r = None

    def __init__(self,en_filename,for_filename,align_filename):
        """
        calls all the methods that create the global variables
        """
        print "create english wordlist..."
        start = time.clock()
        self.english_wordlist,_ = self.create_wordlist(en_filename)
        print "finsished in "+str(time.clock()-start)+"s"
        print "length english dict: "+ str(len(self.english_wordlist))
        
        print "create foreign wordlist..."
        start = time.clock()
        self.foreign_wordlist,self.total_unigrams = self.create_wordlist(for_filename)
        print "finsished in "+str(time.clock()-start)+"s"
        print "length foreing dict: "+ str(len(self.foreign_wordlist))
        
        print "create foreign trigrams..."
        start = time.clock()
        self.foreign_trigrams = self.create_contextlist(1,for_filename)
        print "finsished in "+str(time.clock()-start)+"s"
        print "length foreing trigrams: "+ str(len(self.foreign_trigrams))
        
        print "create trie for foreign pentagrams..."
        start = time.clock()
        self.foreign_penta_trie = self.create_penta_tries(for_filename)
        print "finsished in "+str(time.clock()-start)+"s"
        
        print "create trigram vectors..."
        start = time.clock()
        # self.foreign_trigram_vectors = self.create_feature_vectors(
        #     self.foreign_trigrams,self.foreign_penta_trie,self.total_unigrams)
        self.foreign_trigram_vectors = self.create_random_vectors(
            self.foreign_trigrams,128)
        print "finsished in "+str(time.clock()-start)+"s"
        
        print "create weight matrix..."
        #start = time.clock()
        #self.w = self.create_weights(self.foreign_trigram_vectors)
        #print "finsished in "+str(time.clock()-start)+"s"
        
        print "create allignment matrix..."
        start = time.clock()
        self.a = self.create_alignments(en_filename,for_filename,align_filename,
            self.english_wordlist,self.foreign_wordlist)
        print "finsished in "+str(time.clock()-start)+"s"

        print "writing trie to file..."
        pickle.dump(self.foreign_penta_trie,open("foreign_penta_50000.p","wb"))
        
        #test if left context right word works
        keys = self.foreign_penta_trie.keys()
        print self.foreign_trigrams[0]
        feature = self.create_dict_for_vertex(self.foreign_trigrams[0],keys)
        print "DONE!"

    def create_wordlist(self,filename):
        """
        create a list of all the unique tokens in the given file
        return the list and total number of tokens
        """
        count = 0
        words = set()
        with open(filename) as f:
            for i,line in enumerate(f):
                if i >= self.max_lines:
                    break
                #line = line.lower()
                w = line.split()
                words |= set(w)
                count += len(w)
        return list(words),count

    def create_contextlist(self,n,filename):
        """
        create a list of all the unique ngrams where n is the number of tokens 
        to use left and right of each tokencontext 
        return the list and total number of ngrams
        """
        grams = set()
        with open(filename) as f:
            for i,line in enumerate(f):
                if i >= self.max_lines:
                    break
                w = line.split()
                grams |= set(self.context3(n,w))
        return list(grams)

    def create_penta_tries(self,filename):
        """
        Creates a trie that counts the occurances of each ngram up until
        pentagrams. This is stored in a tree like structure
        """
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
                        if key == "":
                            key += g
                        else:
                            key += "/"+g
                        try:
                            tree[key] += 1
                        except KeyError:
                            tree[key] = 1
                i += 1
        return tree

    def create_feature_vectors(self,trigrams,tree,total):
        """
        Creates the vector representation for every trigram
        using the PMI calculation for the 9 different 
        features
        """
        mat = np.zeros(len(trigrams)*9).reshape(len(trigrams),9)
        for i,trigram in enumerate(trigrams):
            mat[i,0] = self.penta_pmi(trigram,tree)
            mat[i,1] = self.tri_pmi(trigram,tree,total)
            mat[i,2] = self.left_pmi(trigram,tree)
            mat[i,3] = self.right_pmi(trigram,tree)
            mat[i,4] = self.center_pmi(trigram,tree,total)
            mat[i,5] = self.no_center_pmi(trigram,tree)
            mat[i,6] = self.lw_rc_pmi(trigram,tree)
            mat[i,7] = self.lc_rw_pmi(trigram,tree)
            mat[i,8] = self.has_suffix(trigram)
        return mat

    def create_random_vectors(self,trigram,dims):
        return np.random.uniform(0,1,len(trigram)*dims).reshape(len(trigram),dims)

    def penta_pmi(self,trigram,tree):
        """
        returns the pmi for the trigram plus context words
        """
        return 1

    def tri_pmi(self,trigram,tree,total):
        """
        returns the pmi for the trigram
        """
        uni = trigram.split("/")
        if tree.has_key(trigram) and tree.has_key(uni[0]) and tree.has_key(uni[1]) and tree.has_key(uni[2]):
            p_tri = tree[trigram]/total
            p_denom = (tree[uni[0]]/total)*(tree[uni[1]]/total)*(tree[uni[2]]/total)
            return np.log(p_tri/p_denom)
        return 0

    def left_pmi(self,trigram,tree):
        """
        returns the pmi for the two words left of the center word
        """
        #figgure out a manner of calculating pmi
        #calculate for all different bigrams with this trigram
        #combine the different pmi's
        return 1

    def right_pmi(self,trigram,tree):
        """
        returns the pmi for the two words right of the center word
        """
        #figgure out a manner of calculating pmi
        #calculate for all different bigrams with this trigram
        #combine the different pmi's
        return 1

    def center_pmi(self,trigram,tree,total):
        """
        returns the pmi for the center word
        probability of center word p(x3)
        """
        #somehow some of the words are not in the tree..
        if tree.has_key(trigram.split("/")[1]):
            return np.log(float(tree[trigram.split("/")[1]])/total)
        print trigram.split("/")[1]
        return 0

    def no_center_pmi(self,trigram,tree):
        """
        returns the pmi for the trigram without the center word
        """
        #figgure out a manner of calculating pmi
        #calculate for all trigrams where X2 and X4 co-occur in X2 Xi X4
        #combine these
        #or
        #threat X2 X4 as a bigram
        return 1

    def lw_rc_pmi(self,trigram,tree):
        """
        returns the pmi score for the left word and right context combination
        """
        #similar problem as no_center_pmi
        return 1

    def lc_rw_pmi(self,trigram,tree):
        """
        returns the pmi score for the left context and right word combination
        """
        #similar problem as no_center_pmi
        return 1

    def has_suffix(self,trigram):
        """
        returns a score for if the word has a suffix or not
        """
        # if word is longer then 2(or 3/4...)
        # take last 2 (or 3/4...) letters
        # calculate probability for this suffix
        #      -> before hand count number of suffixes
        return 1

    def create_weights(self,vectors):
        """
        calculates the weights between all the trigrams using their vectors
        these are returned in a weight matrix
        """
        # vecs = np.array(vectors)
        # w2 = 1.0 - np.dot(vecs/n(vecs,axis=1)[:,None],(vecs/n(vecs,axis=1)[:,None]).T)
        w = np.zeros(len(vectors)*len(vectors)).reshape(len(vectors),len(vectors))
        for i,vec_a in enumerate(vectors):
            for j,vec_b in enumerate(vectors[i:]):
                for key in set(vec_a).intersection(set(vec_b)):
                    w[i,j+i] += vec_a[key] + vec_b[key]
                w[j+i,i] = w[i,j+i]
        return w

    def create_alignments(self,en_filename,for_filename,align_filename,en_wordlist,for_wordlist):
        a = np.zeros(len(en_wordlist)*len(for_wordlist)).reshape(len(en_wordlist),len(for_wordlist))
        with open(en_filename,'r') as fen:
            with open(for_filename,'r') as ffor:
                with open(align_filename,'r') as fal:
                    for k,(en_line,for_line,al_line) in enumerate(zip(fen,ffor,fal)):
                        if k >= self.max_lines:
                            break
                        en_words = en_line.split()
                        for_words = for_line.split()
                        #for_grams = self.context3(1,for_line.split())
                        al_words = al_line.split()
                        for align in al_words:
                            al = align.split('-')
                            i = en_wordlist.index(en_words[int(al[0])])
                            j = for_wordlist.index(for_words[int(al[1])])
                            a[i,j] += 1
        a = a/a.sum(axis=1)[:,None]
        a[a<0.9] = 0
        return a

    def cosine_similarity(self,vecA,vecB):
        """
        calculates the cosine similarity between two vectors
        """
        return np.dot(vecA,vecB)/(np.dot(vecA,vecA)*np.dot(vecB,vecB))

    def context2(self,n, words):
        """
        context with <s> tags added. n is the ammount of context, left and right
        n represents number of tokens needed to take left and right of each word
        if n is out of range, <s> or </s> tags are added
        """
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

    def context3(self,n, words):
        """
        context with <s> tags added. n is the ammount of context, left and right
        n represents number of tokens needed to take left and right of each word
        if n is out of range, <s> or </s> tags are added
        ngrams are represented as a string seperated by a /
        """
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
        grams = []
        for lst in narray:
            key = ""
            for x in lst:
                if key == "":
                    key += x
                else:
                    key += "/"+x
            grams.append(key)
        return grams

#  FEATURE LISTINGS
    def create_dict_for_vertex(self,vertex,keys):
        features = {}
        lwrc = self.lwrc(vertex,keys)
        lcrw = self.lcrw(vertex,keys)
        lc = self.lc(vertex,keys)
        rc = self.rc(vertex,keys)
        tri_c = self.tri_c(vertex,keys)
        c = self.c(vertex,keys)
        penta = self.penta(vertex,keys)
        tri = self.tri(vertex,keys)
        features.update(lwrc)
        features.update(lcrw)
        features.update(lc)
        features.update(rc)
        features.update(tri_c)
        features.update(c)
        features.update(penta)
        features.update(tri)
        return features
    
    # left word, right context
    # returns list of pentagrams for which trigram is lwrc
    def lwrc(self,tri,keys):
        t = tri.split('/')
        r = r'.*?/'+t[0] + r'/.*?/'+ t[1]+ '/' + t[2]
        lwrc = {}
        for k in keys:
            if len(k.split('/')) == 5:
                if re.match(r,k):
                    #pmi = numpy.log(
                    lwrc[k] = self.foreign_penta_trie[k]
        return lwrc
    
    # left context, right word
    # returns list of pentagrams for which trigram is lcrw
    def lcrw(self,tri,keys):
        t = tri.split('/')
        r = t[0] + '/' + t[1]+ r'/.*?/'+ t[2] + '/.*?'
        print r
        lcrw = {}
        for k in keys:
            if len(k.split('/')) == 5:
                if re.match(r,k):
                    lcrw[k] = self.foreign_penta_trie[k]
        return lcrw

    # returns list of bigrams for which second word is first word of trigram
    def lc(self,tri,keys):
        t = tri.split('/')
        r = r'.*?/' + t[0]
        print r
        lc = {}
        for k in keys:
            if len(k.split('/')) == 2:
                if re.match(r,k):
                    lc[k] = self.foreign_penta_trie[k]
        return lc
    
    # returns list of bigrams for which first word is last word of trigram
    def rc(self,tri,keys):
        t = tri.split('/')
        r = t[2] + r'/.*?'
        print r
        rc = {}
        for k in keys:
            if len(k.split('/')) == 2:
                if re.match(r,k):
                    rc[k] = self.foreign_penta_trie[k]
        return rc

    # returns list of trigrams for which center word is same as trigram
    def tri_c(self,tri,keys):
        t = tri.split('/')
        r = t[0] + r'/' + t[2]
        print r
        tri_c = {}
        for k in keys:
            if len(k.split('/')) == 2:
                if re.match(r,k):
                    tri_c[k] = self.foreign_penta_trie[k]
        return tri_c

    #?
    def c(self,tri,keys):
        t = tri.split('/')
        r = r'.*?/' + t[1] + r'/.*?'
        print r
        c = {}
        for k in keys:
            if len(k.split('/')) == 3:
                if re.match(r,k):
                    c[k] = self.foreign_penta_trie[k]
        return c

    #?
    def penta(self,tri,keys):
        r = r'.*?/' + tri + r'/.*?'
        print r
        penta = {}
        for k in keys:
            if len(k.split('/')) == 5:
                if re.match(r,k):
                    penta[k] = self.foreign_penta_trie[k]
        return penta
    
    def tri(self, tri, keys):
        trigram = {}
        trigram[tri] = self.foreign_penta_trie[tri]
        return trigram

#end FEATURE LISTING

# PMI
#    def compute_PMI(self,vertex,features):
#        PMIs = {}
#        for f in features.keys():
#            PMIs[f] = numpy.log(

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
        return self.foreign_wordlist

    def get_english_wordlist(self):
        return self.english_wordlist

    def export_to_dot_file(self):
        ind = np.sort(np.argpartition(self.w,-5,axis=1)[:,-5:],axis=1)
        count = 0
        count2 = 0
        fout = open('graph2.dot','w')
        fout.write("Vertice_Graph {\n")
        for i,row in enumerate(ind):
            x = self.foreign_trigrams[i]
            x = x.replace("\'\'", "*")
            x = x.replace("\"", "*")
            for j in row:
                count2 += 1
                if self.w[i,j] != 0:
                    count += 1
                    y = self.foreign_trigrams[j]
                    y = y.replace("\'\'", "*")
                    y = y.replace("\"", "*")
                    fout.write("\""+y+"\"->\""+x+"\";\n")
        fout.write("}")
        fout.close()
        print "number of edges: "+str(count)+"   instead of "+str(count2)

    # def create_embeddings(self,embedding_dims,wordlist):
    #     emb = np.random.normal(0., 0.01, embedding_dims*len(wordlist))
    #     print emb
    #     return emb.reshape(len(wordlist),embedding_dims)

    # def create_pentagrams(self,filename):
    #     grams = set()
    #     with open(filename) as f:
    #         for line in f:
    #             grams |= set(self.context2(2,line))
    #     return grams

    # def create_pentagram_vectors(self,embeddings,wordlist,pentagrams):
    #     vectors = []
    #     for gram in pentagrams:
    #         emb = []
    #         for word in gram:
    #             vector.append(embeddings[wordlist.index(word)])
    #         vector = emb
    #         vector += get_trigram(emb)
    #         vector += get_context(emb,'l')
    #         vector += get_context(emb,'r')
    #         vector += get_center_word(emb)
    #         vector += get_context_nocenter(emb)
    #         vector += word_context(emb,'l')
    #         vector += word_context(emb,'r')
    #         vector += has_suffix(emb)
    #         vectors.append(vector)
    #     return np.array(vectors)

    # get left and right context words from pentagram
    # def get_context(self,pentagram,lr):
    #     if lr == 'l':
    #         return(pentagram[0:2])
    #     if lr == 'r':
    #         return(pentagram[len(pentagram)-2:len(pentagram)])

    # def get_trigram(self,pentagram):
    #     return([pentagram[1],pentagram[2],pentagram[3]])

    # # get token word
    # def get_center_word(self,pentagram):
    #     return(pentagram[2])

    # # get trigram without token
    # def get_context_nocenter(self,pentagram):
    #     return([pentagram[1],pentagram[3]])

    # # get left/right word and right/left context
    # def word_context(self,pentagram,lr):
    #     if lr =='l':
    #         return([pentagram[1],pentagram[3],pentagram[4]])
    #     if lr == 'r':
    #         return([pentagram[0],pentagram[1],pentagram[3]])

    # # something with suffix
    # #how do we know what the suffixes are for the language
    # #isn't the suffix count not always zero or equal to center word
    # def has_suffix(self,pentagram):
    #     return 0


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
