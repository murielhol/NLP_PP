fsrc = open('data/en_50000','r').readlines()
ftar = open('data/pt_50000','r').readlines()
fsrcout = open('data/en_50000_noempty','w')
ftarout = open('data/pt_50000_noempty','w')
fout = open('data/en-pt_50000','w')

print "if a line is empty, that line is removed with the line before and after it."
empty = set()
for i,(src,tar) in enumerate(zip(fsrc,ftar)):
    if src[:-1] == "" or tar[:-1] == "":
        empty.add(i-1)
        empty.add(i)
        empty.add(i+1)
print "number of removed lines: "+ str(len(empty))
for i,(src,tar) in enumerate(zip(fsrc,ftar)):
    if i not in empty:
        sr = src.split('\n')
        ta = tar.split('\n')
        for s in sr:
            if s != "":
                fout.write(s)
        fout.write(' ||| ')
        for t in ta:
            if t != "":
                fout.write(t)
        fout.write('\n')
        fsrcout.write(src)
        ftarout.write(tar)
fout.close()