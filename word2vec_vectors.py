import gensim
import time
import pickle

sentences = []
gerrit = 0
with open('../data/pt_50000_noempty') as f:
    for line in f:
        gerrit+=1
        sentences += [line.split()]
            #if gerrit > 10:
                #break

start = time.time()
print 'Starting training of word2vec model'
model = gensim.models.Word2Vec(sentences,min_count=1,workers=25,size=128,window=2)
print 'Done training model in'
print str(time.time() - start)
model.save('pt_noempty')
print 'Done saving model'
