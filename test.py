from graph_construction import Graphs
import numpy as np 

graph = Graphs('data/en_50000','data/pt_50000','data/en-pt_50000.forward.align')
words = graph.get_english_wordlist()
for w in words:
    print w+'\n'