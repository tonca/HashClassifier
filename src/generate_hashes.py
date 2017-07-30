import hashlib
import random
import string
from itertools import repeat
import csv

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction import FeatureHasher



def compute_hashes(word):

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


def generate_hashes(set_len):

    hashes = []

    for i in repeat(None, set_len):

        str_len = random.randint(5,20)

        rand_str = lambda n: ''.join([random.choice(string.printable) for i in xrange(n)])
        
        # Now to generate a random string of length 10
        s = rand_str(str_len) 

        hashes = hashes + compute_hashes(s)

    print "---------------------------------------------"
    print "N of generated hashes: %s" % len(hashes)

    hashes = np.array(hashes)

    X = hashes[:,0]
    Y = hashes[:,1]

    # Characters to Indexes
    ch2int = lambda t: ord(t) if len(t)==1 else 0
    vch2int = np.vectorize(ch2int)

    for i in range(X.shape[0]):
        X[i] = vch2int(X[i]) 

    lengths = np.array([len(line) for line in X])
    max_length = np.max(lengths)
    print "---------------------------------------------"
    print "Maximal hash length: %s" % max_length
    # filling tail spaces to uniform array size 
    for i in range(X.shape[0]):
        diff_length = max_length-len(X[i])
        if diff_length != 0 :
            X[i] = np.concatenate((X[i],np.zeros((diff_length), dtype=np.int))).reshape(-1)


    maxes = np.array([np.max(line) for line in X])
    tot_chars = np.max(maxes) + 1
    print "---------------------------------------------"
    print "Total number of charachters: %s" % tot_chars

    print "---------------------------------------------"
    print "Feature matrix size (max_length, n_chars): %s x %s" % (max_length, tot_chars)

    X_oh = np.zeros((X.shape[0],max_length,tot_chars))

    for i in xrange(0,X.shape[0]):
        for ch_i in xrange(0,X[i].shape[0]):
            X_oh[i,ch_i,X[i][ch_i]] = 1


    lb = LabelBinarizer()
    Y_oh = lb.fit_transform(Y)

    print Y_oh

    return X_oh, Y_oh



if __name__ == '__main__':

    N = 5000
    generate_hashes(N)