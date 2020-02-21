from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from unidecode import unidecode
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict


# functions for text pre-processing
def lem_tagmap():
    tag_map = defaultdict(lambda : wordnet.NOUN)
    tag_map['J'] = wordnet.ADJ
    tag_map['V'] = wordnet.VERB
    tag_map['R'] = wordnet.ADV
    return tag_map

def lemmatized(lmtzr, sentence):
    tagmap = lem_tagmap()
    lmtzed_tokens = []
    for token, tag in pos_tag(word_tokenize(sentence)):
        lemma = lmtzr.lemmatize(token, tagmap[tag[0]])
        lmtzed_tokens.append(lemma)
    return lmtzed_tokens

def pre_process(post):
    ''' post: string, individual post under a thread'''
    # convert input corpus to lower case.
    corpus = post.lower()
    # remove non-ascii characters
    corpus = unidecode(corpus)

    # collecting a list of stop words from nltk and punctuation form
    stopset = stopwords.words('english') + list(string.punctuation) #+ user_names

    # remove stop words and punctuations from string.
    corpus = " ".join([w for w in word_tokenize(corpus) if w not in stopset])

    #lemmatize each word
    lemmatizer = WordNetLemmatizer()
    lem_tokens = lemmatized(lemmatizer, corpus)
    return lem_tokens #' '.join(lem_tokens)

def bigram_query_process(query_terms, bigram_phraser):
    query_tokens = pre_process(query_terms)
    query = bigram_phraser[query_tokens]
    #query = ' '.join(bigram_phraser[query_tokens])
    return query



def tokenize_docs(all_docs, include_title=True):

    k = 1 if include_title else 2

    all_posts_tokenized = []
    title_url = []
    group_posts_tokenized = []
    for doc in all_docs:
        title_url.append(doc[-2:])
        group_post = []
        for post in doc[:-k]:  # [:-1] includes title, [:-2] not include title
            processed_post = pre_process(post)
            all_posts_tokenized.append(processed_post)
            group_post.append(processed_post)
        group_posts_tokenized.append(group_post)

    return all_posts_tokenized, group_posts_tokenized #, title_url


def bigram_group_tokens_rephrase(group_posts_tokenized, bigram_phraser):
    bi_group_posts_tokenized = []
    for thread in group_posts_tokenized:
        bi_thread = []
        for sent in thread:
            bi_thread.append(bigram_phraser[sent])
        bi_group_posts_tokenized.append(bi_thread)
    return bi_group_posts_tokenized


# group posts under a thread together, to use to create tfidf dictionary
def join_thread_tokens(group_posts_tokenized):
    group_tokens = []
    for posts_tokens in group_posts_tokenized:
        temp = []
        for post in posts_tokens:
            temp += post
        group_tokens.append(temp)
    return group_tokens

