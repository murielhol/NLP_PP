# NLP Project: Unsupervised Part-Of-Speech tagging with graph based projection 

##### authors: 
- Alexandra Arkut, BSc
- Janosch Haber, BSc
- Muriel Hol, BSc
- Victor Milewski, BSc

##### Supervision:
- Joost Bastings, MSc

---
#### Problem Description
For the course NLP1 of the Master AI at the UvA, we are working on a project about Low Resources. In this course we have to implement a system that solves a language task for a language with low resources. In our project we chose to work on an Unsupervised Part-Of-Speech tagging with a graph based projection. In this we follow the work from D. Das et al.(2011) and A. Subramanya et al. (2010). 

---
#### Overview
- TO-DO
- Data/Data Preparation
- Vertice Representation
- Neighbourhood Graph Construction
- Alignments

---
#### TO-DO
- Finish calculation of all the features
- Implement POS projection
- Implement POS induction
- Test on new data (not used in graph construction, known lables to test accuracy)
- Use a real low resource language (probably Tagalog)
- Write paper
- Make nice figures/plots/graphs

---
#### Data/Data Preparation

In our experiments we use the English-Portugese Europarl corpus P. Koehn (2005, September). This is a biligual corpus. As preprocessing we lowercased and tokenized the data. Some of the lines where empty, when it in one of the languages was merged with a previous or a next sentence, and in the other language it was not. To maintain a correct translation, the line that was empty in one of the two languages, was removed along with the sentence before and after this line. 

We are only using the first 50000 lines from the corpus, to keep the implementation simple and to obtain a representation of low resource language. 

For the foreign language (in this case Portugese) is the text separated into trigrams. We did this by taking for every word one word before and one word after it, this are our Vertices. If it was the first or the last word in the sentence a start token (<s>) or an end token (</s>) was added. 

We labeled the English side of the corpus with the twelve universal POS tags from S. Petrov et al. (2011).

---
#### Vertice Representation

To create a graph, we first need to represent each foreign Vertice as a vector, so we can do calculations on it. We used the features from A.Subramanya et al. (2010) as listed in the table below:
| Feature                   | Description    |
|--------------------------:|:--------------:|
| Trigram + Context         | x1 x2 x3 x4 x5 |
| Trigram                   | x2 x3 x4       |
| Left Context              | x1 x2          |
| Right Context             | x4 x5          |
| Center Word               | x3             |
| Trigram - Center Word     | x2 x4          |
| Left word + Right Context | x2 x4 x5       |
| Left Context + Right Word | x1 x2 x4       |
| Has Suffix                | has_suffix(x3) |

We did not use all the features. We left the Trigram+context and the Has Suffix feature out. 

We count the number of occurances of each word, and the occurances of a second word following the first, etc. We do this by creating a Trie with pygtrie. So we can efficiently count the occurances of every unigram, bigram, ..., and pentagram. Now that we have the count, we calculate the PMI between the Vertice and every possible feature instantiation. A Vertice will not co-occur with a lot of the features, that is why we only store the ones that occur. 

---
#### Neighbourhood Graph Construction
The neigbourhood Graph is created by calculating the similarities between the Vertices. For each Vertice, relations are creates to the five closes neighbours. 

An optimal solution would be to do a cosine similarity measure. But this would mean we had to create a very long sparce vector. To keep the program simple and avoid memory problems, we calculate the similarity by adding up the PMI values for all the features that appear in both the Vertices. 

---
#### Alignments
To project Part Of Speech tags from english to the foreign language, an alignment is needed between the two languages. First we created alignment files with [fast align](https://github.com/clab/fast_align). We created a forward and backward alignment and then merged these to obtain the optimal results, as was described on their webpage. 

For each of the English words, we counted how often it aligned to each of the words in the foreign language, which was done in a n by m matrix (n: number of english words, m: number of foreign words). For each row we calculated the probabilities of aligning to each of the words. Because we want to use a high confidence alignments, we remove all the probabilities which are lower then 0,9. 

---
#### References
- **Das,   D.,  &  Petrov,   S. (2011).** Unsupervised  part-of-speech tagging  with  bilingual  graph-based  projections. In Proceedings of the 49th annual meeting of the association for computational linguistics:  Human language technologies - volume 1 (pp.   600–609). Stroudsburg,   PA,   USA:   Association  for Computational  Linguistics.
- **Subramanya,   A.,   Petrov,   S.,   &   Pereira,   F. (2010).** Efficient  graph-based  semi-supervised  learning  of  structured tagging   models. In Proceedings  of  the  2010  conference  on  empirical  methods  in  natural  language  processing (pp.   167–176). Stroudsburg,    PA,   USA:   Association   for   Computational Linguistics.
- **Koehn, P. (2005, September).** Europarl: A parallel corpus for statistical machine translation. In MT summit (Vol. 5, pp. 79-86).
- **Petrov, S., Das, D., & McDonald, R. (2011).** A universal part-of-speech tagset. arXiv preprint arXiv:1104.2086.




