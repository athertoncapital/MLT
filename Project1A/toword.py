import sys
dictionary = {1:'e', 2:'t',3:'a',4:'i',5:'n',6:'o',7:'s',8:'h',9:'r',11:'d',12:'l',13:'u',14:'c',15:'m',16:'f',17:'w',18:'y',19:'g',21:'p',22:'b',23:'v',24:'k',25:'q',26:'j',27:'x',28:'z'}
for num in open(sys.argv[1],'r').read().split():
    word = ''
    for n in num.split('0'):
        if n:
            word = word + dictionary.get(int(n))
    print(word)
