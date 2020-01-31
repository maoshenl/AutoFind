import pickle
import numpy as np
import time
import pandas as pd

from gensim.models import Word2Vec


from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from unidecode import unidecode
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict


def load_docs(fname):
    ''' fname: pickle file contains all the posts as lists'''
    all_docs = pickle.load( open(fname, "rb" ) )    
    return all_docs

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

def pre_process(paragraph):
    ''' paragraph: string, individual post under a thread'''
    # convert input corpus to lower case.   
    corpus = paragraph.lower()
    # remove non-ascii characters
    corpus = unidecode(corpus)
                    
    # collecting a list of stop words from nltk and punctuation form
    # string class and create single array.
    stopset = stopwords.words('english') + list(string.punctuation) #+ user_names
                                    
    # remove stop words and punctuations from string.
    corpus = " ".join([w for w in word_tokenize(corpus) if w not in stopset])
                                                    
    #lemmatize each word
    lemmatizer = WordNetLemmatizer()
    lem_tokens = lemmatized(lemmatizer, corpus)
    return ' '.join(lem_tokens)

#prepare corpus for training
''' this result should be pickled '''
def process_doc(all_docs):
    ''' TODO need to process title as well'''
    all_posts = []
    title_url = []
    group_posts_tokenized = []
    for doc in all_docs: 
        title_url.append(doc[-2:])
        group_post = []
        for post in doc[:-2]:
            processed_post = pre_process(post)
            all_posts.append(processed_post)
            group_post.append(word_tokenize(processed_post))

        group_posts_tokenized.append(group_post)
        
    all_posts_tokenized = [word_tokenize(post) for post in all_posts]
    
    return all_posts_tokenized, group_posts_tokenized

def build_model(all_posts_tokenized):
    model = Word2Vec(all_posts_tokenized, min_count=5,size=100, iter=10)

    return model


def count_emb(sentences, model, search_emb, threshold=0.5):
    '''
    Parameters:
        sentences: list, the sentences to comput the embeddings for

        model : `~gensim.models.base_any2vec.BaseAny2VecModel`
            A gensim model that contains the word vectors and the vocabulary

    Returns:
        count of relevent words based on similarity
    '''

    vlookup = model.wv.vocab  # Gives us access to word index and count
    vectors = model.wv        # Gives us access to word vectors
    size = model.vector_size  # Embedding size
                
    #Z = 0
    #for k in vlookup:
    #    Z += vlookup[k].count # Compute the normalization constant Z
    #output = []

    # Iterate all sentences 
    num_accept = 0
    total_count = 0
    for s in sentences:
        # Iterare all words
        for w in s:
            # A word must be present in the vocabulary
            if w in vlookup:
                v = vectors[w]
                if cos_similarity(v, search_emb) > threshold:  
                    num_accept += 1
        total_count += len(s)
    return num_accept/total_count


def sent_emb(sentences, model):
    '''
    Parameters:
        sentences: list, the sentences to comput the embeddings for

        model : `~gensim.models.base_any2vec.BaseAny2VecModel`
            A gensim model that contains the word vectors and the vocabulary

    Returns:
        embedding matrix of dim len(sentences) * dimension
    '''

    vlookup = model.wv.vocab  # Gives us access to word index and count
    vectors = model.wv        # Gives us access to word vectors
    size = model.vector_size  # Embedding size
                
    #Z = 0
    #for k in vlookup:
    #    Z += vlookup[k].count # Compute the normalization constant Z

    output = []
    # Iterate all sentences 
    num_accept = 0
    total_count = 0
    for s in sentences:
        # Iterare all words
        word_count = 0
        v = np.zeros(size, dtype=np.float32)
        for w in s:
            # A word must be present in the vocabulary
            if w in vlookup:
                v += vectors[w]
                word_count += 1
        output.append(v/(word_count+1e-5))

    return np.vstack(output).astype(np.float32)


def first_post_emb(all_docs, model, group_posts_tokenized):
    k = 0
    title_url_emb = []
    
    for i, doc in enumerate(all_docs):
        title0, url = doc[-2:]
        #text_to_emb = [doc[0], doc[-2]]
        try:
            processed_text = [group_posts_tokenized[i][0]] #, group_posts_tokenized[i][-2]
        except:
            print(processed_text, len(processed_text))
        text_emb = [sent_emb(text, model).mean(axis=0) for text in processed_text if len(text)>0]
        avg_text_emb = np.mean(text_emb,axis=0)
    
        title_url_emb.append([title0, url, avg_text_emb])
    return title_url_emb

def search_emb(query, model):
    search = pre_process(query)
    search_emb = sent_emb( [word_tokenize(query)] , model).mean(axis=0)
    return search_emb

def cos_similarity(A,B):
    return np.dot(A,B)/((np.dot(A,A)*np.dot(B,B))**.5 + .001)

def l2(A,B):
    return np.sum(((A-B)*(A-B))**.5)

def score_posts(posts_emb, search_emb):
    #posts emb has [title,url,emb]
    emb_scored = []
    for emb in posts_emb:
        dist_cos = cos_similarity(search_emb, emb[2])
        dist_l2  = l2(search_emb, emb[2])
        emb_scored.append(emb + [dist_cos, dist_l2])
    return emb_scored


# def search and return search results from a title_url_whatever_emb..
def search(search_term, model, posts_emb):
    #posts_emb has title, url and embedding from the model
    query_emb = serach_emb(search_term, model)

    #score the posts
    posts_scored = score_posts(posts_emb, search_emb)
    posts_ranked = sorted(posts_scored, key=lambda x: x[-2], reverse=True)

    #return title and url
    result = [ post[:2]  for i, post in enumerate(posts_ranked)]
    return result























