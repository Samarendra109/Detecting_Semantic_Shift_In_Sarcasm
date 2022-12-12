from tqdm import tqdm

import snscrape.modules.twitter as sntwitter
import pickle

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    count = 1_000_000
    # count = 300_000
    tweets_list2 = []

    for i, tweet in tqdm(enumerate(
            sntwitter.TwitterSearchScraper(
                '(-#irony AND -#sarcasm) -is:retweets -is:replies lang:en'
            ).get_items())):
        if i > count:
            break
        tweets_list2.append(tweet)

    with open("not_sarcasm_data_1M.pkl", "wb") as f:
        pickle.dump(tweets_list2, f)

    # with open("not_sarcasm_data_3M.pkl", "wb") as f:
    #     pickle.dump(tweets_list2, f)

    tweets_list2 = []
    for i, tweet in tqdm(enumerate(
            sntwitter.TwitterSearchScraper(
                '(#irony OR #sarcasm) -is:retweets -is:replies lang:en'
            ).get_items())):
        if i > count:
            break
        tweets_list2.append(tweet)

    with open("sarcasm_data_1M.pkl", "wb") as f:
        pickle.dump(tweets_list2, f)

    # with open("sarcasm_data_3M.pkl", "wb") as f:
    #     pickle.dump(tweets_list2, f)

    print("Hello")
