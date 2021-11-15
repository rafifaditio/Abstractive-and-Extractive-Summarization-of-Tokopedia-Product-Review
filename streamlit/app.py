import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk import tokenize
from nltk.cluster.util import cosine_distance
import networkx as nx

# df = pd.read(csv)

kamus_alay = pd.read_csv('https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv')
dict_alay = pd.Series(kamus_alay.formal.values,index=kamus_alay.slang).to_dict()
dict_alay['brg'] = 'barang'
dict_alay['kw']='tiruan'

diksi = pd.read_csv('diksi_summarization.csv')
diksi = pd.Series(diksi.formal.values,index=diksi.slang).to_dict()

# Make sentiment label based on rating
def sentimen_label(df):
    df['sentiments'] = df['rating'].replace({1:'Negative', 2:'Negative', 3:'Neutral', 4:'Positive', 5:'Positive'})   
    df['sentimen'] = df['rating'].replace({1:0, 2:0, 3:1, 4:2, 5:2})
    return(df)

# Text preprocessing
def clean_text(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"    # emoticons
        u"\U0001F300-\U0001F5FF"    # symbols & pictographs
        u"\U0001F680-\U0001F6FF"    # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"    # flags (iOS)
        u"\U00002500-\U00002BEF"    # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"                   # dingbats
        u"\u3030"
                      "]+", re.UNICODE)

    text = text.lower()                                                             # membuat huruf menjadi kecil
    text = re.sub(emoj, '', text)                                                   # remove emoji
    text = re.sub(r'(.)\1{2,}', r'\1', text)                                        # mengubah huruf berulang diatas 2 kali menjadi 1 saja
    text = re.sub("[0-9]", " ", text)                                               # remove numbers
    text = re.sub("'s", " ", text) 
    text = re.sub("[¹²³¹⁰ⁱ⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ]", "", text)                                 # remove power character
    text = re.sub("[^\w\s]*[_,.!?#&;:><+-/)/(\'\"]", " ", text)                     # remove bukan string dan whitespace
    text = ' '.join([re.sub(r'nya$|ny$', '', i) for i in text.split()])             # menghapus akhir kata 'nya' atau 'ny'
    text = " ".join(dict_alay[w] if w in dict_alay else w for w in text.split())    # replace sesuai dict_alay
    text = " ".join(diksi[w] if w in diksi else w for w in text.split())            # replace sesuai diksi
    text = re.sub(" +", " ", text.strip())                                          # Remove unnecessary white space
    return text

def cleaner(df):
    df['preprocessed'] = df['text'].apply(clean_text)
    return df

def remove_empty(df):
    df['preprocessed'] = df['preprocessed'].replace('',np.nan)
    df.dropna(inplace=True)
    return df

# Group compiling based on sentiment
def compile_sentiment(df,sentiments,resample:False,n):
    if resample:
        grouped = df[df.sentimen==sentiments]['preprocessed'].sample(n)
    else:
        grouped = df[df.sentimen==sentiments]['preprocessed']
    
    n_data = len(grouped)
    joined = '. '.join(list(grouped.values))
    return n_data, joined

# Summarization with Extraction Rule-Based TextRank similarity
def read_text(file):
    input = file.split(". ")
    teks = []
    for kalimat in input:
        teks.append(kalimat.split(" "))
    return teks

def sentence_similarity(sentence_1, sentence_2, stopwords=None):
    if stopwords is None:
        stopwords = []                              # create an empty list to avoid error below
 
    sentence_1 = [w.lower() for w in sentence_1]
    sentence_2 = [w.lower() for w in sentence_2]

    all_words = list(set(sentence_1 + sentence_2))  # create total vocabulary of unique words for the two sentences compared

    vector1 = [0] * len(all_words)                  # prepare one-hot vectors for each sentence over all vocab
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sentence_1:
        if w in stopwords:
            continue 
        vector1[all_words.index(w)] += 1            # list.index(element) returns the index of the given element in the list

    # build the vector for the second sentence
    for w in sentence_2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1-cosine_distance(vector1, vector2)      # Cosine = 0 for similar sentences => returns 1 if perfectly similar

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))  # create a square matrix with dim the num of sentences
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:                                        # ignore if both are same sentences (diagonal of the square matrix)
                continue
            # similarity of each sentence to all other sentences in the text is measured and logged in the matrix
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(file_name, top_n=5, show=False):
    #stop_words = stopwords.words('english')
    stop_words = [] #stopwords.words('indonesian')
    summarize_text = []
    
    # Step 1 - Read text and tokenize
    sentences =  read_text(file_name)
    print("number of sentences in text : ", len(sentences))
    
    # Step 2 - Generate Similary Matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    
    # Step 3 - Rank sentences in similarity matrix. let’s convert the similarity matrix into a graph. 
    # The nodes of this graph will represent the sentences and the edges will represent the similarity scores between
    # the sentences. On this graph, we will apply the PageRank algorithm to arrive at the sentence rankings.
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    # Step 4 - Sort the rank and pick top sentences extract the top N sentences based on their rankings for summary generation
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    if show :
        print("Indexes of top ranked_sentence order are ", ranked_sentence)
    # extract the top N sentences based on their rankings for summary generation
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    
    # Step 5 - Output the summarize text
    print("Summarize Text:\n",".\n".join(summarize_text)+'.')

    teks = ". ".join(summarize_text)
    return tokenize.sent_tokenize(teks)

# make sidebar
st.sidebar.subheader('Upload File')

# setup file upload
file = st.sidebar.file_uploader(label='Please ensure you fullfil all requirements below')

# requirements guide
st.sidebar.write("""
Input File Requirements Guide:

1. Format `.csv`.
2. Consist of 2 columns, formatted sequentially: `text` and `rating`.
3. Column `text` contains review given by customer.
4. Column `rating` contains rating given by customer, in integer format 1 - 5.

""")

# watermark
st.sidebar.caption('Created by [rafifaditio](https://github.com/rafifaditio)')

# title
st.title('Product Review Summarizer App')

# print uploaded file
if file != None:
    df = pd.read_csv(file)
    st.write('Below is your Data:', df)

    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    # preprocess
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df_inf = sentimen_label(df)
    df_inf = cleaner(df_inf)
    df_inf = remove_empty(df_inf)

    # show preprocessed Data
    #st.write('Below is your data after preprocessed:', df_inf[['text', 'rating', 'sentiments', 'preprocessed']])

    # compile based on sentiments label
    #join_neg, join_net, join_pos = compile_sentiment(df_inf)

    # summarize negative sentiments
    st.header('Negative Sentiments')
    st.write('How many Negative Sentiment Reviews would you use?')
    option_neg = st.checkbox('Use all Negative Sentiment Reviews', value=True)

    if option_neg:
        n_neg, join_neg = compile_sentiment(df=df_inf, sentiments=0, resample=False, n=0)
    else:
        maxi = len(df_inf[df_inf['sentimen']==0])
        option_neg = st.number_input('Please Specify a Number: ', min_value=10, max_value=maxi, value=int(maxi*0.3), key='opt_neg')
        st.caption('We recommend to use number of senteces under 150 or you would likely have an error')
        st.caption('System will randomly pick N-numbers reviews from your data to be summarize.')
        n_neg, join_neg = compile_sentiment(df=df_inf, sentiments=0, resample=True, n=option_neg)
    
    st.write('Number of Sentences: ')
    st.write(n_neg)
    neg_sum = generate_summary(join_neg, top_n=5, show=False)
    st.subheader('Summary: ')
    
    for i,t in enumerate(neg_sum):
        st.write(i+1, t)

    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    # summarize neutral sentiments
    st.header('Neutral Sentiments')
    st.write('How many Negative Sentiment Reviews would you use?')
    option_net = st.checkbox('Use all Neutral Sentiment Reviews', value=True)

    if option_net:
        n_net, join_net = compile_sentiment(df=df_inf, sentiments=1, resample=False, n=0)
    else:
        maxi = len(df_inf[df_inf['sentimen']==1])
        option_net = st.number_input('Please Specify a Number: ', min_value=10, max_value=maxi, value=int(maxi*0.3), key='opt_net')
        st.caption('We recommend to use number of senteces under 150 or you would likely have an error')
        st.caption('System will randomly pick N-numbers reviews from your data to be summarize.')
        n_net, join_net = compile_sentiment(df=df_inf, sentiments=1, resample=True, n=option_net)
    
    st.write('Number of Sentences: ')
    st.write(n_net)
    net_sum = generate_summary(join_net, top_n=5, show=False)
    st.subheader('Summary: ')
    
    for i,t in enumerate(net_sum):
        st.write(i+1, t)

    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    # summarize posititive sentiments
    st.header('Positive Sentiments')
    st.write('How many Negative Sentiment Reviews would you use?')
    option_pos = st.checkbox('Use all Positive Sentiment Reviews', value=False)

    if option_pos:
        n_pos, join_pos = compile_sentiment(df=df_inf, sentiments=2, resample=False, n=0)
    else:
        maxi = len(df_inf[df_inf['sentimen']==2])
        option_pos = st.number_input('Please Specify a Number: ', min_value=10, max_value=maxi, value=int(maxi*0.3), key='opt_pos')
        st.caption('We recommend to use number of senteces under 150 or you would likely have an error')
        st.caption('System will randomly pick N-numbers reviews from your data to be summarize.')
        n_pos, join_pos = compile_sentiment(df=df_inf, sentiments=2, resample=True, n=option_pos)
    
    st.write('Number of Sentences: ')
    st.write(n_pos)
    pos_sum = generate_summary(join_pos, top_n=5, show=False)
    st.subheader('Summary: ')
    
    for i,t in enumerate(pos_sum):
        st.write(i+1, t)

    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

else:
    st.write('Please upload file on sidebar.')