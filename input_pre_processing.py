import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# remove stopwords
nltk.download('stopwords')

# example
exception_message = """
Exception in thread "main" java.lang.NullPointerException: Cannot invoke "String.length()" because "str" is null
    at com.example.MyClass.myMethod(MyClass.java:10)
    at com.example.MyClass.main(MyClass.java:5)
"""

# preprocess_Text
def preprocess_text(text):
    # xóa dấu câu
    text = re.sub(r'[^\w\s]', '', text.lower())
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def split_compound_words(words):
    separated_words = []
    for word in words:
        split_camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", word).split()
        for part in split_camel:
            split_parts = re.findall(r"[a-zA-Z]+", part)
            separated_words.extend(split_parts)
    return separated_words

def generate_search_query(exception_message):
    cleaned_text = preprocess_text(exception_message)
    
    # sort 8 keywords
    # in ra score
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    keywords = [feature_names[i] for i in tfidf_scores.argsort()[-8:][::-1]]

    refined_keywords = split_compound_words(keywords)
    refined_keywords = list(dict.fromkeys(refined_keywords))
    
    search_query = " ".join(refined_keywords)
    return search_query

#Testing
exception_message = """
Exception in thread "main" java.lang.NullPointerException: Cannot invoke "String.length()" because "str" is null     at com.example.MyClass.myMethod(MyClass.java:10)     at com.example.MyClass.main(MyClass.java:5)
"""

exception_message2 = """
 exception thread main java lang nullpointerexception cannot invoke string length str null com example myclass mymethod myclass java com example myclass main myclass java
"""
search_query = generate_search_query(exception_message2)
print("Search Query:", search_query)
