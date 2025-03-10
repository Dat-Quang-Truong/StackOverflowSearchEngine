from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import re
import joblib
import time
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import gensim
import smart_open
import warnings
import input_pre_processing
warnings.filterwarnings('ignore')



# create app
import flask
app = Flask(__name__)

################################## Helper functions ######################################################################################

# below mentioned functions are helper functions which is use for text preprocessing
# convert text semantic to proper format
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\n", "", phrase)
    return phrase

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

def remove_stopwords(text):
    '''this function will remove stopwords from text '''
    final_text = ''
    for word in text.split():
        if word not in stopwords:
            final_text += word + ' '
    return final_text

def preprocess_title(text):
    # convert to lower case
    text = text.lower()
    # decontract
    text = decontracted(text)
    # remove all punctuations except a-z and c# and c++
    text = re.sub('[^a-zc#c++]+',' ',text)
    # remove stop words
    text = remove_stopwords(text)
    return text


# load train data
# X_train_limited.cvs trained file dataset.
# X_train = pd.read_csv('X_train.csv')
X_train = joblib.load('X_train.pkl')
print('X_train Loaded') 
# trained gensim w2v model on train data
#tfidf_w2v_vectors_gensim = joblib.load('gensim_tfidf_w2v_vectors_limited.pkl')
tfidf_w2v_vectors_glove = joblib.load('tfidf_w2v_vector_glove.pkl')

print('tfidf_w2v_vectors_gensim loaded')
# Dictionary of words and idf value
tfidf_gensim = joblib.load('tfidf1_final.pkl')
print('tfidf_gensim Loaded')
# trained w2v model using gensim
w2v_model_gensim = joblib.load('gensim_w2v_model_final.pkl')
print(type(w2v_model_gensim))
print('w2v_model_gensim loaded')
# w2vec words vocabulary
# w2v_words = list(w2v_model_gensim.wv.vocab)
w2v_words = list(w2v_model_gensim.wv.index_to_key) 
# tfidf words
tfidf_words = set(tfidf_gensim.get_feature_names_out())
# dictionary of tfidf and idf values
dictionary = dict(zip(tfidf_gensim.get_feature_names_out(),tfidf_gensim.idf_))

with open('glove_vectors', 'rb') as f:
    model = pickle.load(f)
    glove_words = set(model.keys())


######################################### route paths ##################################################


#@app.route('/')
#def hello_world():
#    return 'Hello World!'


@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def final_func(top_n=10):
    ''' This function will find top similar result for given query using gensim w2v'''
    start = time.time()
    # initialize  vector for user query
    main_vec = np.zeros(300)
    # initialize tfidf weight
    weight_sum = 0
    # get text from user
    to_predict_list = request.form.to_dict()
    # preprocess question
    # text = preprocess_title(to_predict_list['user_question'])
    # text = input_pre_processing.generate_search_query(to_predict_list['user_question'])
    text = preprocess_title(text)
    print("text input: ", text)
    #splitting the sentence
    text_list = list(text.split())
    for word in text_list:
        #finding if word is present in tfidf and in w2v words
        if word in tfidf_words and word in w2v_words :
            #finding vector of word from glove model
            vect = model[word]
            #compute tfidf
            tf_idf = dictionary[word]*(text_list.count(word)/len(text_list)) 
            # adding vector * tfidf to main_vec
            main_vec+= (vect*tf_idf)
            # summing tfidf values
            weight_sum += tf_idf
    if weight_sum !=0:
        # devide by weight_sum
        main_vec /= weight_sum
    # find cosine similarity
    # tfidf word2vec have trained using gensim
    similarities =  cosine_similarity((main_vec).reshape(1, -1), Y=tfidf_w2v_vectors_glove, dense_output=True)
    # sort similarities 
    sort = np.argsort(similarities[0])
    # get top similarity indices  in descending order
    similarity_index = np.array(list(reversed(sort)))
    # find top n similarities
    top_similarity_index = similarity_index[:top_n]
    # print top similarity values
    print('Top cosine similarities are ======>',similarities[0][top_similarity_index])
    similar_questions_title = X_train['title'][top_similarity_index]
    similar_answers = X_train['preprocessed_answer'][top_similarity_index]
    base_url = "https://stackoverflow.com/questions/"
    similar_questions_url = [f"{base_url}{qid}" for qid in X_train['id'][top_similarity_index]]
    
    
    # add log to file dataset
    data = {
        'input': to_predict_list['user_question'],
        'title': similar_questions_title,
        'answers': similar_answers,
        'url': similar_questions_url
    }
    df = pd.DataFrame(data)
    log_file = 'model1_logs.csv'

    # Kiểm tra nếu file tồn tại và có dữ liệu
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        try:
            existing_df = pd.read_csv(log_file)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            updated_df = df  # Nếu file bị rỗng, ghi lại từ đầu
    else:
        updated_df = df

    # Luôn luôn ghi lại dữ liệu sau khi cập nhật
    updated_df.to_csv(log_file, index=False)
    print(f"Data saved to {log_file}")



    # similar_questions_url = X_train['id'][top_similarity_index]
    total_time = (time.time() - start)
    print('\t')
    print('Total time ===========> ',total_time)
    dictt = {'top_10' : list(zip(similar_questions_title,similar_questions_url))}
    #return jsonify({'Top 10 Similar questions using gensim w2v': list(zip(similar_questions_title,similar_questions_url))})
    return flask.render_template('response.html',res = dictt )


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8081, debug=True)

