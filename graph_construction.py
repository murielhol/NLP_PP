# Graph-based POS-tagging for low resource languages
# Part 1: Graph Construction

# Alexandra Arkut, Janosch Haber, Muriel Hol, Victor Milewski
# v.0.01 - 2016-11-23

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
        self.foreign_trigram_vectors = self.create_trigram_vectors()
        print "finsished in "+str(time.clock()-start)+"s"
        
        print "create weight matrix..."
        start = time.clock()
        self.w = self.create_weights(self.foreign_trigram_vectors)
        print "finsished in "+str(time.clock()-start)+"s"
        
        print "create allignment matrix..."
        start = time.clock()
        self.a = self.create_alignments(en_filename,for_filename,align_filename,
            self.english_wordlist,self.foreign_wordlist)
        print "finsished in "+str(time.clock()-start)+"s"

        print "writing trie to file..."
        pickle.dump(self.foreign_penta_trie,open("foreign_penta_50000.p","wb"))
        print
        #test if left context right word works
#        keys = self.foreign_penta_trie.keys()
#        print self.foreign_trigrams[0]
#        feature = self.tri_features(self.foreign_trigrams[0],keys)
#        print feature

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
                print grams
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
#                        print key
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
        for key_a in vectors.keys():
            i = self.foreign_trigrams.index(key_a)
            for key_b in vectors.keys():
                j = self.foreign_trigrams.index(key_b)
                if w[i,j] == 0:
                    for key in set(vectors[key_a].keys()).intersection(set(vectors[key_b].keys())):
                        w[i,j] += vectors[key_a][key] + vectors[key_b][key]
                    w[j,i] = w[i,j]
        return w

    def create_alignments(self,en_filename,for_filename,align_filename,en_wordlist,for_wordlist):
#        a = np.zeros(len(en_wordlist)*len(for_wordlist)).reshape(len(en_wordlist),len(for_wordlist))
        a = np.zeros(len(en_wordlist)*len(self.foreign_trigrams)).reshape(len(self.foreign_trigrams),len(en_wordlist))
        with open(en_filename,'r') as fen:
            with open(for_filename,'r') as ffor:
                with open(align_filename,'r') as fal:
                    for k,(en_line,for_line,al_line) in enumerate(zip(fen,ffor,fal)):
                        if k >= self.max_lines:
                            break
                        en_words = en_line.split()
                        #for_words = for_line.split()
                        for_grams = self.context3(1,for_line.split())
                        al_words = al_line.split()
                        for align in al_words:
                            al = align.split('-')
                            j = en_wordlist.index(en_words[int(al[0])])
                            i = self.foreign_trigrams.index(for_grams[int(al[1])])
                            #j = for_wordlist.index(for_words[int(al[1])])
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
                    gram += ['<s>']
            gram += words[i-n+len(gram):i]
            if i+n > len(words)-1:
                gram += words[i:len(words)]
                for _ in range(2*n+1-len(gram)):
                    gram += ['<|s>']
            else:
                gram += words[i:i+n+1]
            
        #make additional shorter ngrams if last word
            if i == len(words)-1:
                for j in np.arange(n):
                    gram2 = []
                    gram2 += words[i-j+len(gram2):i+1]
                    for _ in range(n):
                        gram2 += ['<|s>']
                    narray.append(gram2)
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
                    gram += ['<s>']
            gram += words[i-n+len(gram):i]
            if i+n > len(words)-1:
                gram += words[i:len(words)]
                for _ in range(2*n+1-len(gram)):
                    gram += ['<|s>']
            else:
                gram += words[i:i+n+1]
            #make shorter ngrams if last word
#            if i == len(words)-1:
#                for j in range(n):
#                    gram2 = []
#                    gram2 += words[i-j+len(gram):i]
#                    for _ in range(2*n+1-len(gram)):
#                        gram2 += ['<|s>']
#                    narray.append(gram2)
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
    def create_trigram_vectors(self):
        vectors = {}
        keys = self.foreign_penta_trie.keys()
        for gram in self.foreign_trigrams:
            vector = {}
            features,cooccur = self.create_dict_for_vertex(gram,keys)
            for feature in features.keys():
                try:
                    a = (cooccur[feature]*1.0/7*self.total_unigrams)
                    b = (features[feature]*1.0/7*self.total_unigrams)
                    c = (self.foreign_penta_trie[gram]*1.0/self.total_unigrams)
                    vector[feature] = np.log(a/(b*c))
                except KeyError:
                    vector[feature] = 0
            vectors[gram] = vector
        return vectors
    
    def create_dict_for_vertex(self,vertex,keys):
        features = {}
        cooccur = {}
        uni = self.uni_features(vertex)
        bi = self.bi_features(vertex,keys)
        tri = self.tri_features(vertex,keys)
        uni_co = self.uni_cooccur(vertex)
        bi_co = self.bi_cooccur(vertex,keys)
        tri_co = self.tri_cooccur(vertex,keys)
        features.update(uni)
        features.update(bi)
        features.update(tri)
        cooccur.update(uni_co)
        cooccur.update(bi_co)
        cooccur.update(tri_co)
        return features, cooccur
    
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


# NEW FEATURE LISTING

    def uni_features(self,tri):
        t = tri.split('/')
        dict = {}
        dict[t[1]] = self.foreign_penta_trie[t[1]]
        return dict
    
    def uni_cooccur(self,tri):
        t = tri.split('/')
        dict = {}
        dict[t[1]] = self.foreign_penta_trie[tri]
        return dict
    
    def bi_features(self,tri,keys):
        t = tri.split('/')
        dict = {}
        r1 = r'^([^/]*?' + '/' + t[0]+r')$' # left context
        r2 = r'^('+ t[2] + '/' + r'[^/]*?)$' # right context
        r3 = r'^(' + t[0] + '/' + r'[^/]*?' + '/' + t[2] + r')$' # tri - center
        for k in keys:
            if (re.match(r1,k) or re.match(r2,k)):# and len(k.split('/')) == 2 :
                    try:
                        dict[k] += self.foreign_penta_trie[k]
                    except KeyError:
                        dict[k] = self.foreign_penta_trie[k]
            if re.match(r3,k): #and len(k.split('/')) == 3:
                    k2 = k.split('/')
                    k3 = str(k2[0])+'/'+str(k2[2])
                    try:
                        dict[k3] += self.foreign_penta_trie[k]
                    except KeyError:
                        dict[k3] = self.foreign_penta_trie[k]
        return dict
    
    def bi_cooccur(self,tri,keys):
        t = tri.split('/')
        dict = {}
        r1 = r'^([^/]*?' + '/' + tri+r')$' # left context
        r2 = r'^('+ tri + '/' + r'[^/]*?)$' # right context
        r3 = r'^(' + t[0] + '/' + t[1] + '/' + t[2] + r')$' # tri - center
        for k in keys:
            if (re.match(r1,k) or re.match(r2,k)):# and len(k.split('/')) == 2 :
                try:
                    dict[k] += self.foreign_penta_trie[k]
                except KeyError:
                    dict[k] = self.foreign_penta_trie[k]
            if re.match(r3,k): #and len(k.split('/')) == 3:
                k2 = k.split('/')
                k3 = str(k2[0])+'/'+str(k2[2])
                try:
                    dict[k3] += self.foreign_penta_trie[k]
                except KeyError:
                    dict[k3] = self.foreign_penta_trie[k]
        return dict

    def tri_features(self,tri,keys):
        t = tri.split('/')
        dict = {}
        r1 = r'^'+ tri + r'$'# trigram
        r2 = r'^([^/]*?/'+ t[0] + r'/[^/]*?/[^/]*?)$' # left context, right word
        r3 = r'^(' + t[0] + '/' + r'[^/]*?' + '/' + t[2] + r'/[^/]*?)$' # left word, right context
        for k in keys:
            if (re.match(r1,k)):
                    try:
                        dict[k] += self.foreign_penta_trie[k]
                    except KeyError:
                        dict[k] = self.foreign_penta_trie[k]
            if (re.match(r2,k)):
                    k2 = k.split('/')
                    k3 = str(k2[0])+'/'+str(k2[1]) + '/' + str(k2[3])
                    try:
                        dict[k3] += self.foreign_penta_trie[k]
                    except KeyError:
                        dict[k3] = self.foreign_penta_trie[k]
            if (re.match(r3,k)):
                    k2 = k.split('/')
                    k3 = str(k2[0])+'/'+str(k2[2]) + '/' + str(k2[3])
                    try:
                        dict[k3] += self.foreign_penta_trie[k]
                    except KeyError:
                        dict[k3] = self.foreign_penta_trie[k]
        return dict
    
    def tri_cooccur(self,tri,keys):
        t = tri.split('/')
        dict = {}
        r1 = r'^'+ tri + r'$'# trigram
        r2 = r'^([^/]*?/'+ tri +')$' # left context, right word
        r3 = r'^(' + tri + r'/[^/]*?)$' # left word, right context
        for k in keys:
            if (re.match(r1,k)):
                try:
                    dict[k] += self.foreign_penta_trie[k]
                except KeyError:
                    dict[k] = self.foreign_penta_trie[k]
            if (re.match(r2,k)):
                k2 = k.split('/')
                k3 = str(k2[0])+'/'+str(k2[1]) + '/' + str(k2[3])
                try:
                    dict[k3] += self.foreign_penta_trie[k]
                except KeyError:
                    dict[k3] = self.foreign_penta_trie[k]
            if (re.match(r3,k)):
                k2 = k.split('/')
                k3 = str(k2[0])+'/'+str(k2[2]) + '/' + str(k2[3])
                try:
                    dict[k3] += self.foreign_penta_trie[k]
                except KeyError:
                    dict[k3] = self.foreign_penta_trie[k]
        return dict
    
    
# END NEW FEATURE LISTING

    def get_weight_matrix_all(self):
        return self.w

    def get_weight_matrix(self,n):
        ind = np.argpartition(self.w,-n,axis=1)[:,-n:]
        w2 = np.zeros(np.shape(self.w))
        for i in range(np.shape(self.w)[0]):
            for j in ind[i]:
                w2[i,j] = self.w[i,j]
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
