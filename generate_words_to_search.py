# Generates a list of sentiment words to check in the corpus

from nltk.corpus import wordnet as wn

from senti_classifier.senti_classifier import synsets_scores


if __name__ == '__main__':
    word_list = []
    for w in synsets_scores:
        # Finding words with high positive score or high negative score.
        if (synsets_scores[w]['pos'] == 1.0) or (synsets_scores[w]['neg'] == 1.0):
            word_list.extend([l.name() for l in wn.synset(w).lemmas()])

    print(set(word_list))