import sys,math,numpy
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

x = []
for word in bagofwords:
    row = []
    for doc in docwords:
        counter = Counter(doc)
        count_t = counter.get(word)
        if count_t == None:
            count_t = 0
        max_w = counter.most_common()[0][1]
        tf = float(count_t)/max_w
        count = freq.get(word)
        idf = math.log(float(len(docwords))/count)
        row.append(tf*idf)
    x.append(row)
Y = numpy.array(x)
print 'filename, closest match, cosine'
l = len(sys.argv[1:])
count = 0
for q in range(0, l): 
    cos_max = 0
    doc_max = 1
    for v in range(0, l):
        if q != v:
            cos = (numpy.dot(Y[:,q],Y[:,v]))/(numpy.linalg.norm(Y[:,q]) * numpy.linalg.norm(Y[:,v]))
            if cos > cos_max:
                cos_max = cos
                doc_max = v + 1
    doc = sys.argv[q + 1]
    match = sys.argv[doc_max]
    if doc.startswith('good') and match.startswith('good') or doc.startswith('bad') and match.startswith('bad'):
        count = count + 1
    print doc + ', ' + match + ', ' + str(cos_max)
print 'Overall score: ' + str(float(count*100)/l) + '%'
            
                
                

