from gensim import corpora, models, similarities
from gensim.models import Phrases
from gensim.models.phrases import Phraser


# sentence_stream is the first output from tokenize_docs in text_preprocess.py
def get_bigram_phraser(sentence_stream, min_count=5, threshold=10):
    #sentence_stream = all_posts_tokenized_title[:]
    bigram = Phrases(sentence_stream, min_count=min_count, threshold=threshold)
    bigram_phraser = Phraser(bigram)
    return bigram_phraser

# all_thread_tokens is output from join_thread_tokens(group_posts_tokenized) function in text_preprocess.py
def tfidf_resources(all_thread_tokens):
    #make dictionary of bigram token
    dictionary = corpora.Dictionary(all_thread_tokens)
    #Convert the word into vector, and now you use tfidf model
    corpus = [dictionary.doc2bow(text) for text in all_thread_tokens]
    tfidf_model = models.TfidfModel(corpus)
    tfidf_index = similarities.SparseMatrixSimilarity(tfidf_model[corpus], num_features = len(dictionary.token2id))

    return (dictionary, tfidf_model, tfidf_index)

