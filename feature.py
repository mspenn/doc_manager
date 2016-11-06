# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from constant import DIR_CACHE_FEATURE, CHAR_ENDL, CHAR_SPACE

__author__ = 'Zak Penn'


class Feature:
    def __init__(self):
        self.freq = None
        self.tfidf = None
        self.feature_names = None
        self.feature_select = None
        self.feature_label = None

    def make_vsm(self, corpus_data):
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        self.freq = vectorizer.fit_transform(corpus_data)
        self.tfidf = transformer.fit_transform(self.freq)
        self.feature_names = vectorizer.get_feature_names()

    def reduce_feature(self, corpus_label, n=0, k=-1):
        # descending sort and choose top 5
        weight_sum = self.tfidf.sum(axis=0)
        if n == 0:
            # auto select features
            sum_max = np.max(weight_sum)
            sum_min = np.min(weight_sum)
            # sum_var = np.var(weight_sum)
            quarter = sum_min + (sum_max - sum_min) / 4
            feature_index = np.where(weight_sum > quarter)[1].getA1()
            freq_select = self.freq[:, feature_index]
        elif n > 0:
            feature_weight = np.argsort(-weight_sum)
            feature_index = feature_weight[:, :n].getA1()
            freq_select = self.freq[:, feature_index]
        else:
            # use all features
            feature_index = range(0, len(self.feature_names))
            freq_select = self.freq

        # clamp values, for chi-square test
        freq_select[freq_select > 1] = 1
        feature_vec = self.tfidf[:, feature_index]

        if k == 0:
            k_best = int(len(feature_index) * 0.9)
        elif k < 0:
            k_best = len(feature_index)
        else:
            k_best = int(k)
        # chi-square test
        ch2 = SelectKBest(chi2, k=k_best)
        self.feature_label = np.array(corpus_label)
        self.feature_select = ch2.fit_transform(freq_select, self.feature_label)
        self.feature_vec = feature_vec[:, self.feature_select.indices]

    def print_vsm(self):
        print "Shape: ", self.tfidf.shape
        weight = self.tfidf.toarray()
        for i in range(len(weight)):
            print "[%3d] TF-IDF Weight" % i
            for j in range(len(self.feature_names)):
                print self.feature_names[j], weight[i][j]

    def store(self, feature_id):
        features_path = "%s/%s" % (DIR_CACHE_FEATURE, feature_id)
        out = open(features_path, "w")
        rows, cols = self.feature_select.shape
        for r in xrange(rows):
            out.write("%d" % self.feature_label[r])
            out.write(CHAR_SPACE)
            for c in xrange(cols):
                out.write("%d:%f " % (c, self.feature_vec[r, c]))
            out.write(CHAR_ENDL)
        out.close()
