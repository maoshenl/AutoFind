import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
import numpy as np
import bigram_tfidf as btf
import helper
import text_preprocess as tp

all_docs = pickle.load( open('all_docs.p', "rb" ) )
bi_phraser, bi_dict, bi_tfidf, bi_tfidf_index = pickle.load(open('bi_tfidf_package.p','rb') )
dictionary, tfidf, tfidf_index = pickle.load(open('tfidf_package.p','rb') )

st.title('C-RV Owners Club  - Problem and Issues')
query = st.text_input("search your issues", '')

search_model = btf.TFIDFSearch(all_docs, bi_dict, bi_tfidf, bi_tfidf_index, bigram_phraser = bi_phraser)
query_results = search_model.tfidf_search(query)


if query !='':
    show_n = 200
    for i, result in helper.paginator("go to page" , query_results[:show_n],  on_sidebar=True):
        title_url = result[:2]
        post0 = result[-1]
        text = str(i+1) + '.  ['+title_url[0]+']' + '(' + title_url[1] +')'
        st.markdown('<font size="4">' + text + '</font>', unsafe_allow_html=True) 
        #st.markdown('<br>'+ post0[:100]+'<br>', unsafe_allow_html=True)
        st.markdown('<p style="text-indent: 30px">'+ post0[:250]+ ' ...'+'</p>', unsafe_allow_html=True)
        #st.markdown('<blockquote>'+ post0[:200]+ '</blockquote>', unsafe_allow_html=True)
        #<br> are you one of them, we have a curated list of chillout tracks<br>












