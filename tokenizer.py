# -*- coding: utf-8 -*-
from abc import abstractmethod
from collections import Counter
import os
from string import maketrans, punctuation
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from nltk.probability import *
from gnosis.indexer import TextSplitter
from constant import DOC_LANG, DIR_CACHE_TOKEN, CHAR_SPACE


__author__ = 'Zak Penn'


class Tokenizer:
    def __init__(self):
        pass

    @abstractmethod
    def get_tokens(self, text, n=-1):
        pass

    @abstractmethod
    def get_word(self, data):
        pass

    def store(self, category_name, id, data):
        dir_name = "%s/%s" % (DIR_CACHE_TOKEN, category_name)
        if (not os.path.exists(dir_name)) or (not os.path.isdir(dir_name)):
            os.makedirs(dir_name)
        out = open("%s/%s" % (dir_name, id), "w")
        for d in data:
            out.write(self.get_word(d))
            out.write(CHAR_SPACE)
        out.close()

    @staticmethod
    def load_category(category_name):
        data = []
        category_path = os.path.join(DIR_CACHE_TOKEN, category_name)
        if not os.path.exists(category_path):
            return data
        files_list = os.listdir(category_path)
        for filename in files_list:
            data.append(open(os.path.join(category_path, filename)).read())
        return data


class nltkTokenizer(Tokenizer):
    def get_tokens(self, text, n=-1):
        # remove the punctuation using the character deletion step of translate
        # make translate table
        blank = " " * len(punctuation)
        no_punctuation = text.lower().translate(maketrans(punctuation, blank))
        tokens = word_tokenize(no_punctuation)
        filtered = [w for w in tokens if not w in stopwords.words(DOC_LANG)]
        stemmed = self.stem_tokens(filtered)
        # count = Counter(filtered)
        # print (count.most_common(10))
        count = Counter(stemmed)
        # print (count.most_common(n))
        # return count
        if n == -1: return count
        return count.most_common(n)

    def stem_tokens(self, tokens, stemmer=PorterStemmer()):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def get_word(self, data):
        return data


class gnosisTokenizer(Tokenizer):
    # pre-process document, and output word tokens
    def get_tokens(self, text, n=-1):
        article = TextSplitter().text_splitter(text)
        stemmer = PorterStemmer()
        stems = FreqDist()
        for word in article:
            lower_word = word.lower()
            # filter the stop words
            if not lower_word in stopwords.words(DOC_LANG):
                stems.inc(stemmer.stem_word(lower_word))
        # top n frequency items
        if n == -1: return stems.items()
        return stems.items()[:n]

    def get_word(self, data):
        return (data[0] + " ") * data[1]