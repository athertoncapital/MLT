import sys
dictionary = {'e':1, 't':2,'a':3,'i':4,'n':5,'o':6,'s':7,'h':8,'r':9,'d':11,'l':12,'u':13,'c':14,'m':15,'f':16,'w':17,'y':18,'g':19,'p':21,'b':22,'v':23,'k':24,'q':25,'j':26,'x':27,'z':28}
output = open(sys.argv[2],'w')
for word in open(sys.argv[1],'r').read().split(' '):
    for char in list(word):
        if char.isalpha():
            output.write(str(dictionary.get(char.lower()))+'0')
    output.write('\n')
