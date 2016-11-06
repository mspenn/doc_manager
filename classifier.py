# -*- coding: utf-8 -*-
from abc import abstractmethod
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from constant import DIR_CACHE_FEATURE, CHAR_ENDL
from svmutil import *

__author__ = 'Zak Penn'


class Classifier:
    def __init__(self):
        pass

    @abstractmethod
    def do_classify(self):
        pass

    @staticmethod
    def predict_info(name, actual, predict, average='macro'):
        m_precision = metrics.precision_score(actual, predict, average=average)
        m_accuracy = metrics.accuracy_score(actual, predict)
        m_recall = metrics.recall_score(actual, predict, average=average)
        print 'predict info: %s' % name
        print 'accuracy:{0:.5f}'.format(m_accuracy)
        print 'precision:{0:.5f}'.format(m_precision)
        print 'recall:{0:0.5f}'.format(m_recall)
        print 'f1-score:{0:.5f}'.format(metrics.f1_score(actual, predict, average=average))


class LibSvmClassifier(Classifier):
    def __init__(self, feature_id):
        self.feature_id = feature_id

    def do_classify(self):
        skf = StratifiedKFold(n_splits=2)
        y, x = svm_read_problem("%s/%s" % (DIR_CACHE_FEATURE, self.feature_id))
        y_np = np.array(y)
        x_np = np.array(x)
        y_predict = np.copy(y_np)
        for train_index, test_index in skf.split(x_np, y_np):
            # m = svm_train(y[:50], x[:50], '-c 4')
            # p_label, p_acc, p_val = svm_predict(y[50:], x[50:], m)
            # choose linear kernel function -t 0
            m = svm_train(y_np[train_index].tolist(), x_np[train_index].tolist(), '-c 16 -t 0')
            p_label, p_acc, p_val = svm_predict(y_np[test_index].tolist(), x_np[test_index].tolist(), m)
            y_predict[test_index] = p_label
            print y_np[test_index].tolist(), CHAR_ENDL, p_label, CHAR_ENDL, p_acc, CHAR_ENDL, p_val
        return y_np, y_predict


class SvmClassifier(Classifier):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def do_classify(self):
        skf = StratifiedKFold(n_splits=2)
        y_predict = np.copy(self.y)
        for train_index, test_index in skf.split(self.x, self.y):
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            # default with 'rbf'
            # kernel='linear'
            svclf = SVC(kernel='linear').fit(x_train, y_train)
            y_predict[test_index] = svclf.predict(x_test)
        return self.y, y_predict


class NaiveBayesClassifier(Classifier):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def do_classify(self):
        skf = StratifiedKFold(n_splits=2)
        y_predict = np.copy(self.y)
        for train_index, test_index in skf.split(self.x, self.y):
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            clf = MultinomialNB().fit(x_train, y_train)
            y_predict[test_index] = clf.predict(x_test)
        return self.y, y_predict
