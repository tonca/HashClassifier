import hashlib
import random
import string
from itertools import repeat
import csv

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher


print('helo')

set_len = 1000

def generate_hashes(word):

    gen_hash = []

    HASH_FUNCTIONS = {
        'md5' : hashlib.md5, 
        'sha1' : hashlib.sha1, 
        'sha224' : hashlib.sha224, 
        'sha256' : hashlib.sha256, 
        'sha384' : hashlib.sha384, 
        'sha512' : hashlib.sha512
    }

    for h_f in HASH_FUNCTIONS:
        m = HASH_FUNCTIONS[h_f]()
        m.update(word)
        
        hashed = list(m.digest())
        gen_hash.append([hashed, h_f])

    return gen_hash


hashes = []

for i in repeat(None, set_len):

    str_len = random.randint(5,20)

    rand_str = lambda n: ''.join([random.choice(string.printable) for i in xrange(n)])
    
    # Now to generate a random string of length 10
    s = rand_str(str_len) 

    # print s
    # print "------------------------------------------"
    hashes = hashes + generate_hashes(s)

print len(hashes)

hashes = np.array(hashes)

print hashes.shape
print hashes
X = hashes[:,0]
Y = hashes[:,1]

print X.shape
print X[0]
ch2int = lambda t: ord(t) if len(t)==1 else 0

vch2int = np.vectorize(ch2int)
print vch2int(X[0])
for i in range(X.shape[0]):
    X[i] = vch2int(X[i])

# label_enc = LabelEncoder()
# label_enc.fit(X[:])  

# string_int = label_enc.transform(X[:])
lengths = np.array([len(line) for line in X])
max_length = np.max(lengths)

# filling holes
for i in range(X.shape[0]):
    diff_length = max_length-len(X[i])
    if diff_length != 0 :
        X[i] = np.concatenate((X[i],np.zeros((diff_length), dtype=np.int))).reshape(-1)

print X[0]
maxes = np.array([np.max(line) for line in X])
max_max = np.max(maxes)
print max_max

X_oh = np.zeros((X.shape[0],64,256))

for i in xrange(0,X.shape[0]):
    for ch_i in xrange(0,X[i].shape[0]):
        X_oh[i,ch_i,X[i][ch_i]] = 1

print X_oh

