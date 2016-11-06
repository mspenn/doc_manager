# -*- coding: utf-8 -*-
from classifier import Classifier, SvmClassifier, NaiveBayesClassifier, LibSvmClassifier
from corpus import Corpus
from feature import Feature
from tokenizer import Tokenizer

__author__ = 'Zak Penn'

if __name__ == "__main__":
    data_path = "data/20_newsgroups"
    corpus = Corpus()
    corpus.load_index(data_path)
    # gen tokens, uncomment when it needs regenerate
    # corpus.gen_tokens(gnosisTokenizer())
    # calc features
    corpus_tokens = []
    corpus_labels = []
    for category in corpus.category_list:
        content = Tokenizer.load_category(category)
        if content:
            corpus_tokens.extend(content)
            corpus_labels.extend([corpus.category_list.index(category)] * len(content))
    feature = Feature()
    feature.make_vsm(corpus_tokens)
    # feature.print_vsm()
    # reduce feature, k==0 means auto detect
    # feature.reducex(corpus_labels, cate_list=corpus.category_list)
    feature.reduce_feature(corpus_labels, k=0)
    feature_id = "feature.txt"
    feature.store(feature_id)

    # classify
    # lib svm
    classifier = LibSvmClassifier(feature_id)
    y_actual, y_predict = classifier.do_classify()
    Classifier.predict_info("Lib SVM", y_actual, y_predict)
    #  sklearn svm
    classifier = SvmClassifier(feature.feature_vec, feature.feature_label)
    y_actual, y_predict = classifier.do_classify()
    Classifier.predict_info("Sklearn SVM", y_actual, y_predict)
    # naive bayes
    classifier = NaiveBayesClassifier(feature.feature_vec, feature.feature_label)
    y_actual, y_predict = classifier.do_classify()
    Classifier.predict_info("Naive Bayes", y_actual, y_predict)