# Graph-based POS-tagging for low resource languages
# Part 1: Graph Construction

# Alexandra Arkut, Janosch Haber, Muriel Hol, Victor Milewski
# v.0.01 - 2016-11-23

from collections import Counter
import numpy as np
import sys
 
# FEATURES
pentagram_count = Counter()
trigram_count = Counter()
left_context_count = Counter()
right_context_count = Counter()
center_count = Counter()
trigram_nocenter_count = Counter()
lw_rc_count = Counter()
rw_lc_count = Counter()

# not used
def ngram(n, words):
    narray = [None]*(len(words)+1-n)
    for i in range(len(words)+1-n):
        narray[i] = [words[i:i+n]]
    return(narray)

# context with <s> tags added. n is the ammount of context, left and right
def context2(n, words):
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

# context with <s> tags added
def context(n, words):
    narray = [None]*(len(words))
    for i in range(len(words)):
        narray[i] = [words[i-n:i+n+1]]
        if i-n < 0:
            narray[i] = ['<S>']+words[i:i+n+1]
        elif i+n > len(words)-1:
            narray[i] = words[i-n:len(words)]+['</S>']
            return(narray)
    return(narray)

# get left and right context words from pentagram
def get_context(pentagram,lr):
    if lr == 'l':
        return(pentagram[0:2])
    if lr == 'r':
        return(pentagram[len(pentagram)-2:len(pentagram)])

def get_trigram(pentagram):
    return([pentagram[1],pentagram[2],pentagram[3]])

# get token word
def get_center_word(pentagram):
    return(pentagram[2])

# get trigram without token
def get_context_nocenter(pentagram):
    return([pentagram[1],pentagram[3]])

# get left/right word and right/left context
def word_context(pentagram,lr):
    if lr =='l':
        return([pentagram[1],pentagram[3],pentagram[4]])
    if lr == 'r':
        return([pentagram[0],pentagram[1],pentagram[3]])

# something with suffix
#how do we know what the suffixes are for the language
#isn't the suffix count not always zero or equal to center word
def has_suffix(pentagram):
    return 0

def cosine_similarity(vecA,vecB):
    return np.dot(vecA,vecB)/(np.dot(vecA,vecA)*np.dot(vecB,vecB))

def create_counters(filename):
    #corpus = open(filename,'r')
    with open(filename) as f:
        for line in f:
        #for line in corpus.readline():
            line = line.lower()
            w = line.split()
            pentas = context2(2,w) # call this with tags added so that double tags appear
            #pentagrams_portuguese += penta
            for gram in pentas:
                pentagram_count[str(gram)] += 1.
                trigram_count[str(get_trigram(gram))]+= 1.
                left_context_count[str(get_context(gram,'l'))] += 1.
                right_context_count[str(get_context(gram,'r'))] += 1.
                center_count[str(get_center_word(gram))] += 1.
                trigram_nocenter_count[str(get_context_nocenter(gram))] += 1.
                lw_rc_count[str(word_context(gram,'l'))] += 1.
                rw_lc_count[str(word_context(gram,'r'))] += 1.

def create_gram_features():
    #create all the feature vectors from the counts
    grams = []
    for gram,penta_count in pentagram_count.most_common():
        feature_vec = []
        feature_vec.append(penta_count)
        feature_vec.append(trigram_count[str(get_trigram(gram))])
        feature_vec.append(left_context_count[str(get_context(gram,'l'))])
        feature_vec.append(right_context_count[str(get_context(gram,'r'))])
        feature_vec.append(center_count[str(get_center_word(gram))])
        feature_vec.append(trigram_nocenter_count[str(get_context_nocenter(gram))])
        feature_vec.append(lw_rc_count[str(word_context(gram,'l'))])
        feature_vec.append(rw_lc_count[str(word_context(gram,'r'))])
        feature_vec.append(has_suffix(gram))
        grams.append((gram,feature_vec))
    return grams


def create_weight_matrix(gram_features):
    w = np.zeros(len(gram_features)**2)
    w = w.reshape(len(gram_features),len(gram_features))
    for i,(gram,vecA) in enumerate(gram_features):
        for j,(_,vecB) in enumerate(gram_features):
            w[i,j] = cosine_similarity(vecA,vecB)
    return w

def get_weight_matrix(filename):
    create_counters(filename)
    gram_features = create_gram_features()
    print len(gram_features)
    return create_weight_matrix(gram_features)

