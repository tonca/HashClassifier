import hashlib
import random
import string
from itertools import repeat
import csv

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.utils import shuffle


HASH_FUNCTIONS = ['whirlpool','sha512']

def compute_hashes(word):

    gen_hash = []

    h_f = random.choice(HASH_FUNCTIONS)
    m = hashlib.new(h_f)
    m.update(word)
    
    hashed = list(m.digest())
    gen_hash.append([hashed, h_f])

    return gen_hash


def generate_hashes(set_len):

    hashes = []

    for i in repeat(None, set_len):

        str_len = random.randint(8,30)

        rand_str = lambda n: ''.join([random.choice(string.printable) for i in xrange(n)])
        
        # Now to generate a random string of length 10
        s = rand_str(str_len) 

        hashes = hashes + compute_hashes(s)

    # print "---------------------------------------------"
    # print "N of generated hashes: %s" % len(hashes)

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
    min_length = np.min(lengths)

    # print "---------------------------------------------"
    # print "Maximal hash length: %s" % max_length
    
    # filling tail spaces to uniform array size 
    X_fill = np.copy(X)
    for i in range(X.shape[0]):
        diff_length = max_length-len(X[i])
        if diff_length != 0 :
            X_fill[i] = np.concatenate((X[i],np.zeros((diff_length), dtype=np.int))).reshape(-1)


    maxes = np.array([np.max(line) for line in X])
    tot_chars = np.max(maxes) + 1

    # print "---------------------------------------------"
    # print "Total number of charachters: %s" % tot_chars

    # print "---------------------------------------------"
    # print "Feature matrix size (max_length, n_chars): %s x %s" % (max_length, tot_chars)
    tot_chars = 256
    X_oh = np.zeros((X.shape[0],max_length,tot_chars))

    for i in xrange(0,X.shape[0]):
        for ch_i in xrange(0,X[i].shape[0]):
            X_oh[i,ch_i,X[i][ch_i]] = 1

    Y_oh = np.zeros((Y.shape[0],len(HASH_FUNCTIONS)))
    for i in xrange(0,Y.shape[0]):
        class_id = HASH_FUNCTIONS.index(Y[i])
        Y_oh[i,class_id] = 1
    
    # print Y_oh
    # print X.shape
    # print X_fill.shape

    X_oh, Y_oh = shuffle(X_oh,Y_oh)

    return X_oh, Y_oh


def hash_generator(batch_size):
    
    while 1:
        features, labels = generate_hashes(batch_size)
        yield features, labels


if __name__ == '__main__':

    print hashlib.algorithms_available
    N = 1000
    print generate_hashes(N)

    hasher = hashlib.new('whirlpool')
    hasher.update("word")
    
    hashed = list(hasher.digest())
    print hashed