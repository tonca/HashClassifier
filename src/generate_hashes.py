import hashlib
import random
import string
from itertools import repeat
import csv

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



print('helo')

set_len = 100

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
        
        hashed = m.digest()
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

X = list(X)

label_enc = LabelEncoder()
label_enc.fit(X)  

string_int = label_enc.transform(X)

print np.max(string_int)