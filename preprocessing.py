from nltk.tokenize import StanfordTokenizer as st

fout = open('europarl-v7.pt-en.tok.pt','w')
with open('europarl-v7.pt-en.pt') as f:
            for line in f:
                wrds = st().tokenize(line)
                for w in wrds:
                    fout.write(w+' ')
                fout.write('\n')
fout.close()
fout = open('europarl-v7.pt-en.tok.en','w')
with open('europarl-v7.pt-en.en') as f:
            for line in f:
                wrds = st().tokenize(line)
                for w in wrds:
                    fout.write(w+' ')
                fout.write('\n')
fout.close()