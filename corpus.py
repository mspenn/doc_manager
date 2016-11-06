# -*- coding: utf-8 -*-
import os

__author__ = 'Zak Penn'


class Corpus:
    def __init__(self):
        self.category_list = []
        self.data = {}
        self.data_path = None

    def load_index(self, path):
        self.data_path = path
        self.category_list = os.listdir(path)
        for category in self.category_list:
            category_path = os.path.join(path, category)
            self.data[category] = os.listdir(category_path)
        return self.data

    def gen_tokens(self, tokenizer):
        for category in self.data:
            for file in self.data[category]:
                self.gen_token(tokenizer, category, file)

    def gen_token(self, tokenizer, category_name, file_name):
        text = open(os.path.join(self.data_path, category_name, file_name)).read()
        tokens = tokenizer.get_tokens(text)
        tokenizer.store(category_name, file_name, tokens)



