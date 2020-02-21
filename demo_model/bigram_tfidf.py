import text_preprocess as tp


class TFIDFSearch:
    def __init__(self, all_threads, dictionary, tfidf_model, index, bigram_phraser=None):

        self.all_docs = all_threads
        self.dict = dictionary
        self.tfidf_model = tfidf_model
        self.index = index
        self.bigram_phraser = bigram_phraser


    def tfidf_search(self, query):

        if self.bigram_phraser:
            query = tp.bigram_query_process(query, self.bigram_phraser)
        else:
            query = tp.pre_process(query)

        #print('--', query, '---')
        title_url = [doc[-2:]+[doc[0]]  for doc in self.all_docs]
        kw_vector = self.dict.doc2bow(query)
        #feature_cnt = len(self.dict.token2id)

        sim = self.index[self.tfidf_model[kw_vector]]
        thread_ranked = [x for _, x in sorted(zip(sim, title_url), reverse=True)]
        return thread_ranked


