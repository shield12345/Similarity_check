import pickle
import enchant
import re
import nltk
from tokenizer import custom_tokenizer
import os

d = enchant.Dict('en_US')
clitics = set(["'s", "n't", "'ll", "'d", "'m", "'ve", "'re"])


def check_irregularity(token):
    return any(c.isalpha() for c in token) and not token in clitics and not d.check(token)


def get_irregular_token_stats():
    if not os.path.exists('pickles/tokens.pkl'):
        print('Please run tokenizer.py to generate pickled file. Exiting...')
        exit(0)

    with open('pickles/tokens.pkl', 'rb') as f:
        tokens = pickle.load(f)

    irregular_tokens = {}
    for token in tokens:
        if check_irregularity(token):
            irregular_tokens[token] = irregular_tokens.get(token, 0) + 1

    frequent_tokens = sorted(irregular_tokens, key=irregular_tokens.__getitem__, reverse=True)[:20]
    with open('tokenization/custom_tokenizer/irregular_tokens_stats.txt', 'w') as f:
        for token in frequent_tokens:
            f.write(str(token) + ': ' + str(irregular_tokens[token]) + '\n')


def pos_tag_sentences():
    if not os.path.exists('pickles/posts.pkl'):
        print('Please run extract_clean_posts.py to generate pickled file. Exiting...')
        exit(0)
    with open('pickles/posts.pkl', 'rb') as f:
        posts = pickle.load(f)

    sentences = []
    pos_tags = {}

    for post in posts:
        post = re.sub(r'<code>.*</code>', '', post)  # remove inline code snippets

        for paragraph in post.split('\n'):
            if len(paragraph) == 0:
                continue

            nltk_sentences = nltk.tokenize.sent_tokenize(paragraph)
            for nltk_sentence in nltk_sentences:
                words = nltk_sentence.split()
                for word in words:
                    if check_irregularity(word):
                        sentences.append(nltk_sentence)
                        break

        if len(sentences) >= 10:
            break

    for sentence in sentences:
        if len(pos_tags) == 10:
            break

        tokens = []
        for token in custom_tokenizer(sentence):
            if token:
                tokens.append(token)
        pos_tags[sentence] = nltk.pos_tag(tokens)

    with open('tokenization/custom_tokenizer/sentences_pos_tags.txt', 'w') as f:
        for key in list(pos_tags.keys()):
            f.write(str(key) + '\n')
            f.write(str(pos_tags[key]) + '\n')
            f.write('\n----\n' + '\n')

    if not os.path.exists('pickles/'):
        os.makedirs('pickles/')

    with open('pickles/pos_tags_custom.pkl', 'wb') as f:
        pickle.dump(pos_tags, f)


get_irregular_token_stats()
pos_tag_sentences()
