from gensim.models import KeyedVectors


def test_similarity_score(word_to_index, word_vec_array, training_indices = None):
    lsa_model = KeyedVectors(50)
    word_list = [item[0].upper() for item in sorted(word_to_index.items(), key=lambda item: item[1])]
    lsa_model.add_vectors(word_list, word_vec_array)
    rejected_words = []

    count, total = 0, 0
    with open('MEN_dataset_lemma_form_full.txt', "r") as f, open('MEN_dataset_lemma_form_full1.txt', "w") as f1:
        while line := f.readline():
            w1, w2, score = line.split()
            w1 = w1.split('-')[0]
            w2 = w2.split('-')[0]
            if w1 in word_to_index and w2 in word_to_index:
                if (training_indices is None) or \
                        ((word_to_index[w1] not in training_indices) and (word_to_index[w2] not in training_indices)):
                    count += 1
                    f1.write(w1 + " " + w2 + " " + score + "\n")
                elif training_indices is not None:
                    if word_to_index[w1] in training_indices:
                        rejected_words.append(w1)
                    if word_to_index[w2] in training_indices:
                        rejected_words.append(w2)

            total += 1

    p_corr, tmp, ratio = lsa_model.evaluate_word_pairs('MEN_dataset_lemma_form_full1.txt', delimiter=' ')
    print(p_corr)
    if rejected_words:
        print("Rejected Words: ", set(rejected_words))

    return p_corr[0]