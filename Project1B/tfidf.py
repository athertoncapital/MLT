import sys,math
from collections import Counter
bagofwords = []
def createbag(doc):
    words = []
    global bagofwords
    for text in open(doc,'r').read().split(' '):
        word = ''.join(c for c in text.lower() if c.isalpha())
        if word != '':
            if word not in bagofwords:
                bagofwords.append(word)
            words.append(word)
    return words
docwords = []
for doc in sys.argv[1:]:
    docwords.append(createbag(doc))
freq = {}
for word in bagofwords:
    counter = 0
    for n in range(0, len(docwords)):
        if word in docwords[n]:
            counter = counter + 1
    freq[word] = counter
print 'term, ' + ', '.join(doc for doc in sys.argv[1:])
for word in bagofwords:
    row = [word]
    for doc in docwords:
        counter = Counter(doc)
        count_t = counter.get(word)
        if count_t == None:
            count_t = 0
        max_w = counter.most_common()[0][1]
        tf = float(count_t)/max_w
        count = freq.get(word)
        idf = math.log(float(len(docwords))/count)
        row.append(str(tf*idf))
    print ', '.join(row) 
