import argparse
import os
import pickle
import re
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from os.path import exists
from pathlib import Path
from string import punctuation
from typing import List

import nltk
import torch
from nltk import word_tokenize
from tqdm import tqdm

import gensim.downloader
import numpy as np

from nltk.corpus import stopwords

punctuation = set(list(punctuation))
sarcasm_split = 'hxab'
sarcasm_key = "sarcabcd"


def get_sarc_postfix():
    return sarcasm_split + sarcasm_key


# Uncomment the below to train the word vectors for stop words
words_to_search = set(stopwords.words('english'))

# Uncomment the below to train the word vectors for sentiment words
# words_to_search = {'abject', 'unsound', 'splendid', 'homologic', 'sad', 'bounder',
#                    'henpecked', 'first-rater', 'like', 'good', 'heel', 'misguide',
#                    'mislead', 'abduction', 'shoddy', 'cloud_nine', 'unsurpassable',
#                    'cad', 'blissfulness', 'kudos', 'first-class', 'top-hole', 'unfit',
#                    'pitiful', 'walking_on_air', 'discourtesy', 'honorable', 'dominated',
#                    'deplorable', 'cheapjack', 'dog', 'motormouth', 'research_worker',
#                    'lead_astray', 'topping', 'fantabulous', 'anger', 'top-flight',
#                    'blackguard', 'congratulations', 'disrespect', 'researcher', 'felicity',
#                    'estimable', 'scrimy', 'bliss', 'sorry', 'excellent', 'balmy', 'happiness',
#                    'enjoy', 'angriness', 'mean', 'wonderfulness', 'seventh_heaven', 'lamentable',
#                    'scut_work', 'soft', 'admirableness', 'misdirect', 'extolment', 'investigator',
#                    'shitwork', 'admirability', 'tawdry', 'love', 'unfortunate', 'bad',
#                    'worst', 'praise', 'distressing', 'hound', 'mild', 'respectable', 'homological', 'sensational'}


class GloveTweetVectors:

    def __init__(self):
        if exists("glove_vector.pkl"):
            print("Reading from pickle")
            with open("glove_vector.pkl", "rb") as f:
                self.gv = pickle.load(f)
        else:
            self.gv = gensim.downloader.load('glove-twitter-50')
            with open("glove_vector.pkl", "wb") as f:
                pickle.dump(self.gv, f)
        self.zero_vector = np.zeros_like(self.gv.get_vector('is'))
        pass

    def get_vocab(self):
        return list(self.gv.index_to_key)

    def get_word_to_vec_dict(self, words):
        word_to_vec_dict = {}
        for word in tqdm(words):
            if self.gv.has_index_for(word):
                word_to_vec_dict[word] = self.gv.get_vector(word)
            else:
                word_comps = word.split(sarcasm_split)
                if (len(word_comps) == 2) and (word_comps[1] == sarcasm_key):
                    word_to_vec_dict[word] = self.gv.get_vector(word_comps[0])
                else:
                    word_to_vec_dict[word] = self.zero_vector

        return word_to_vec_dict


class ContextWeight(Enum):
    NORMAL = 0
    LOG = 1
    SQRT = 2

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return ContextWeight[s.upper()]
        except KeyError:
            return s


@dataclass
class Utils:
    context_size: int
    data_file: str
    word2index_file: str
    frequency_file: str
    context_weight: ContextWeight

    def __init__(self, context_size=10, context_weight=ContextWeight.NORMAL):
        self.context_size = context_size
        self.context_weight = context_weight
        self.file_prefix = f"Test__C{context_size}_CW_{self.context_weight.name}/"
        Path(self.file_prefix).mkdir(parents=True, exist_ok=True)
        self.data_file = self.file_prefix + "training_data.pkl"
        self.word2index_file = self.file_prefix + "word2index.pkl"
        self.frequency_file = self.file_prefix + "word_frequency.pkl"
        self.glove_vec_instance = GloveTweetVectors()
        self.top_search_file = self.file_prefix + "top_search_words.pkl"

    def get_word2index(self):
        with open(self.word2index_file, "rb") as f:
            word2index = pickle.load(f)
        return word2index

    def get_wordfrequency(self):
        with open(self.frequency_file, "rb") as f:
            word_frequency = pickle.load(f)
        return word_frequency

    def get_top_search_words(self):
        with open(self.top_search_file, 'rb') as f:
            top_search_words = pickle.load(f)
        return top_search_words

    def get_context_windows(self, sentences):

        context_window_list = []
        max_window_size = 0
        for sent in tqdm(sentences):
            sent = [word.lower() for word in sent if (word.isalpha()) and (word not in punctuation)]
            for i, word in enumerate(sent[:-1]):
                c_window = tuple(sent[i:min(len(sent), i + self.context_size + 1)])
                if len(c_window) > max_window_size:
                    max_window_size = len(c_window)
                context_window_list.append(c_window)

        print("Max Context Window Size", max_window_size) # Should be self.context_size + 1
        return context_window_list

    def get_context_weight(self, j):
        if self.context_weight == ContextWeight.NORMAL:
            return 1 / j
        elif self.context_weight == ContextWeight.LOG:
            return 1 / (1 + np.log10(j))
        elif self.context_weight == ContextWeight.SQRT:
            return 1 / (np.sqrt(j))

    def get_data(self, force=False):

        if force or not (exists(self.data_file) and exists(self.word2index_file)):

            sents = get_not_sarcasm_data()

            f_words, most_common_words = get_freq_and_top_words(sents)
            top_search_words = get_top_16_words_from_words_to_search(f_words)
            print(top_search_words)

            with open(self.top_search_file, "wb") as f:
                pickle.dump(top_search_words, f)

            # This code is to maintain the training words are in both the database
            for word in words_to_search:
                if word in most_common_words:
                    most_common_words.append(word + get_sarc_postfix())

            sarcasm_sents = get_sarcasm_data(top_search_words)
            s_f_words, s_most_common_words = get_freq_and_top_words(sarcasm_sents)

            # This code is to maintain the training words are in both the database
            for word in words_to_search:
                if word in most_common_words:
                    s_most_common_words.append(word)

            f_words.update(s_f_words)
            print(len(f_words))

            with open(self.frequency_file, "wb") as f:
                pickle.dump(f_words, f)

            most_common_words = set(most_common_words).intersection(s_most_common_words)

            final_words = most_common_words
            word2index = {w: i for i, w in enumerate(final_words)}
            print(len(final_words))

            indices = OrderedDict()
            values = []

            context_windows_list = self.get_context_windows(sents)
            print(len(context_windows_list))

            for context_window in tqdm(context_windows_list):

                center_word = context_window[0]

                if center_word not in word2index:
                    continue

                for j, context_word in enumerate(context_window):

                    if (context_word in word2index) and (context_word != center_word):
                        index_of_words = (word2index[center_word], word2index[context_word])
                        if index_of_words not in indices:
                            indices[index_of_words] = len(values)
                            values.append(0)
                        values[indices[index_of_words]] += self.get_context_weight(j)

            context_windows_list = self.get_context_windows(sarcasm_sents)
            print(len(context_windows_list))

            for context_window in tqdm(context_windows_list):

                center_word = context_window[0]

                if center_word not in word2index:
                    continue

                for j, context_word in enumerate(context_window):

                    if (context_word in word2index) and (context_word != center_word):
                        index_of_words = (word2index[center_word], word2index[context_word])
                        if index_of_words not in indices:
                            indices[index_of_words] = len(values)
                            values.append(0)
                        values[indices[index_of_words]] += self.get_context_weight(j)

            torch_indices = torch.LongTensor(list(indices.keys()))
            torch_values = torch.tensor(values)

            with open(self.data_file, "wb") as f:
                pickle.dump((torch_indices, torch_values), f)

            with open(self.word2index_file, "wb") as f:
                pickle.dump(word2index, f)

        else:
            with open(self.data_file, "rb") as f:
                torch_indices, torch_values = pickle.load(f)

        print("Entries", torch_values.size())
        return torch_indices, torch_values

    def get_training_indices(self):
        word_frequency = self.get_wordfrequency()
        word2index = self.get_word2index()

        training_words = self.get_top_search_words()
        training_words.extend([word+get_sarc_postfix() for word in training_words])

        training_indices = np.zeros(len(word2index), dtype=bool)
        for word in training_words:
            training_indices[word2index[word]] = True

        return torch.BoolTensor(training_indices)

    def get_glove_vectors(self):
        word2index = self.get_word2index()

        sorted_w2i = sorted(word2index.items(), key=lambda tup: tup[1])
        sorted_wordlist = [tup[0] for tup in sorted_w2i]

        w2v_dict = self.glove_vec_instance.get_word_to_vec_dict(sorted_wordlist)
        glove_vectors = [w2v_dict[word] for word in sorted_wordlist]

        return torch.FloatTensor(glove_vectors)

    def get_glove_vectors_and_train_indices(self):
        return self.get_glove_vectors(), self.get_training_indices()


def get_top_16_words_from_words_to_search(word_frequency) -> List[str]:
    word_f_list = []
    for word in words_to_search:
        if word in word_frequency:
            word_f_list.append((word, word_frequency[word]))
    training_words = sorted(word_f_list, key=lambda tup: tup[1], reverse=True)[:16]
    return [tup[0] for tup in training_words]


def get_freq_and_top_words(sents):
    words = [word for sent in sents for word in sent]
    words = [word.lower() for word in words]
    words = [word for word in words if word.isalpha()]
    words = [word for word in words if word not in punctuation]

    f_words = nltk.FreqDist(words)

    tmp_list = [t for t in f_words.most_common(70000) if t[1] >= 10]
    most_common = sorted(tmp_list, key=lambda tup: tup[1])
    most_common_words = [tup[0] for tup in most_common]

    return f_words, most_common_words


def remove_irony_hashtag(sentence):
    pattern_irony = re.compile("#irony", re.IGNORECASE)
    sentence = pattern_irony.sub("", sentence)
    pattern_sarcasm = re.compile("#sarcasm", re.IGNORECASE)
    sentence = pattern_sarcasm.sub("", sentence)
    return sentence


def get_not_sarcasm_data():
    # 3M is 300_000 not 3Million.
    with open("not_sarcasm_data_3M.pkl", "rb") as f:
        sentences = pickle.load(f)

    return tweets_pre_processing([sent.content for sent in sentences])


def get_sarcasm_data(top_search_words):
    # 3M is 300_000 not 3Million.
    with open("sarcasm_data_3M.pkl", "rb") as f:
        sentences = pickle.load(f)

    return tweets_pre_processing([sent.content for sent in sentences], top_search_words)


def switch_top_search_words(sentence, top_search_words):

    def get_switch(word):
        if word.lower() in top_search_words:
            return word + get_sarc_postfix()
        else:
            return word

    return [get_switch(word) for word in sentence]


def tweets_pre_processing(sentences: List[str], top_search_words=None):
    """

    We remove URLs
    We remove usernames (starting with @)
    We remove #irony and #sarcasm
    We remove the hash and just keep the tag words
    All non-alphabetical characters except for 0 are removed

    Reference:
    https://www3.tuhh.de/sts/hoou/data-quality-explored/3-2-simple-transf.html

    :param top_search_words: If the words are present then add these words with a different key
    :param sentences:
    :return: preprocessed sentences
    """

    if top_search_words is None:
        top_search_words = []

    def camel_case_split(identifier):
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    result_sentences = []
    top_search_words = set(top_search_words) if top_search_words else {}
    for i, sentence in tqdm(enumerate(sentences)):
        sentence = re.sub(r'https?://[^ ]+', '', sentence)
        sentence = re.sub(r'@[^ ]+', '', sentence)
        sentence = remove_irony_hashtag(sentence)
        sentence = re.sub(r'#', '', sentence)
        sentence = re.sub(r'([A-Za-z])\1{2,}', r'\1', sentence)
        sentence = re.sub(r' 0 ', 'zero', sentence)
        sentence = re.sub(r'[^A-Za-z ]', '', sentence)
        sentence = word_tokenize(sentence)
        sentence = [
            word for multi_word in sentence
            for word in camel_case_split(multi_word)
        ]
        sentence = switch_top_search_words(sentence, top_search_words)
        result_sentences.append(sentence)

    return result_sentences


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=int, help="Context Window", default=10)
    parser.add_argument("--w", type=ContextWeight.argparse, help="Context Weight",
                        choices=list(ContextWeight), default='NORMAL')
    return parser.parse_args()


def generate_data():
    # combinations = (
    #     (10, ContextWeight.NORMAL), (10, ContextWeight.LOG), (10, ContextWeight.SQRT),
    #     (15, ContextWeight.NORMAL), (15, ContextWeight.LOG), (15, ContextWeight.SQRT),
    #     (20, ContextWeight.NORMAL), (20, ContextWeight.LOG), (20, ContextWeight.SQRT),
    # )

    combinations = (
        (20, ContextWeight.LOG),
    )
    for c in combinations:
        util_i = Utils(context_size=c[0], context_weight=c[1])
        util_i.get_data(force=True)


if __name__ == '__main__':
    generate_data()
