# -*- coding: utf-8 -*-
import math

__author__ = 'Zak Penn'

# just simple implementation

def tf(word, count):
    return float(count[word]) / sum(count.values())


def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)


def idf(word, count_list):
    return math.log(float(len(count_list)) / (1 + n_containing(word, count_list)))


def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


class Tfidf:
    def calc_tfidf(self, cnt_list):
        for i, count in enumerate(cnt_list):
            print("Top words in document {}".format(i + 1))
            scores = {word: tfidf(word, count, cnt_list) for word in count}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:3]:
                print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))


                #test
                # corpus_list = [test2(TEXT1, 50), test2(TEXT2, 50), test2(TEXT3, 50)]
                # for i, corpus in enumerate(corpus_list):
                #     print("Top words in document {}".format(i + 1))
                #     scores = {word: tfidf(word, corpus, corpus_list) for word in corpus}
                #     sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                #     for word, score in sorted_words[:10]:
                #         print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
                # print test1(TEXT, 50)
                # print  test2(TEXT)