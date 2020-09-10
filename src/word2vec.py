import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from adjustText import adjust_text

def word2vec(og_df, columnName, tdif, word_embedding_size=300, train_algo=0):
    df = og_df.copy()
    # Grab all experience
    all_columns = df[columnName]

    # Create a list of strings, one for each title
    entry_list = [entry for entry in all_columns]

#     # Collapse the list of strings into a single long string for processing
#     big_entry_list = ' '.join(entry_list)

#     # Tokenize the string into words
#     tokens = word_tokenize(big_entry_list)

#     # Remove non-alphabetic tokens, such as punctuation
#     words = [word.lower() for word in tokens if word.isalpha()]

#     # Filter out stopwords
#     stop_words = set(stopwords.words('english'))
#     words = [word for word in words if not word in stop_words]
    
    # create custom word2vec model/word embeddings
    # return dataframe of words & word2vec vector representations
    model = create_model(df, columnName, word_embedding_size, train_algo)
    
    # adding responses into corpus
    corpus = [preprocess(entry) for entry in entry_list]

    # Remove docs that don't include any words in W2V's vocab
    corpus, entry_list = filter_docs(corpus, entry_list, lambda doc: has_vector_representation(model, doc))
    
    # Filter out any empty docs
    corpus, entry_list = filter_docs(corpus, entry_list, lambda doc: (len(doc) != 0))
    x = []
    
    # tdif or no tdif 
    if tdif:
        tfid_corpus = []
        for l in corpus:
            tfid_corpus.append(' '.join(l))

        vectorizer = TfidfVectorizer()
        vecs = vectorizer.fit_transform(tfid_corpus)
        
        tfidf_dict = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        
        for doc in corpus: # append the vector for each document
            x.append(document_vector_tdif(model, doc, tfidf_dict))
    else:
        for doc in corpus: # append the vector for each document
            x.append(document_vector(model, doc))
    
    
    X = np.array(x) # list to array

    df_w_vectors = pd.DataFrame(x)

    df_w_vectors[columnName] = entry_list
    # Use pd.concat to match original experiences with their vectors
    main_w_vectors = pd.concat((df_w_vectors, df), axis=1)

    # Get rid of vectors that couldn't be matched with the main_df
    main_w_vectors.dropna(axis=0, inplace=True)
    
    return main_w_vectors
    
def create_model(df, columnName, word_embedding_size=300, train_algo=0):
    experiences = df[columnName].str.split().tolist()
    model = Word2Vec(experiences, size=word_embedding_size, sg=train_algo)
    return model
#     words = list(model.wv.vocab)

#     # Filter the list of vectors to include only those that Word2Vec has a vector for
#     # vector_list = [model[word] for word in words if word in model.vocab]
#     vector_list = [model[word] for word in words if word in model.wv.vocab]

#     # Create a list of the words corresponding to these vectors
#     # words_filtered = [word for word in words if word in model.vocab]
#     words_filtered = [word for word in words if word in model.wv.vocab]

#     # Zip the words together with their vector representations
#     word_vec_zip = zip(words_filtered, vector_list)

#     # Cast to a dict so we can turn it into a DataFrame
#     word_vec_dict = dict(word_vec_zip)
#     word2vec_df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    
#     return word2vec_df

    
# preprocessing functions to keep experience response intact
# https://github.com/sdimi/average-word2vec/blob/master/notebook.ipynb
def document_vector_tdif(model, doc, tfidf_dict):
    """
    return average of all vector representation within text blurb, with weighted words using tdif 
    """
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in model.wv.vocab]
    weighted = []
    for word in doc:
        if word in tfidf_dict:
            weighted.append(tfidf_dict[word] * model[word])
        else:
            weighted.append(model[word])
    return np.mean(weighted, axis = 0)

def document_vector(model, doc):
    """
    return average of all vector representation within text blurb
    """
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in model.wv.vocab]
    return np.mean(model[doc], axis=0)


# Our earlier preprocessing was done when we were dealing only with word vectors
# Here, we need each document to remain a document 
def preprocess(text):
    """
    process each text blurb as a text blurb, not just word vectors
    """
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] 
    return doc

# Function that will help us drop documents that have no word vectors in word2vec
def has_vector_representation(model, doc):
    """
    check if at least one word of the document is in the
    word2vec dictionary
    """
#     return not all(word not in word2vec_model.vocab for word in doc)
    return not all(word not in model.wv.vocab for word in doc)

# Filter out documents
def filter_docs(corpus, texts, condition_on_doc):
    """
    Filter corpus and texts given the function condition_on_doc which takes a doc. The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts)

def load_model(df, columnName):
    # Grab all entries
    all_entry = df[columnName]

    # Create a list of strings, one for each entry
    entry_list = [entry for entry in all_entry]

    # Collapse the list of strings into a single long string for processing
    big_entry_string = ' '.join(entry_list)

    # Tokenize the string into words
    tokens = word_tokenize(big_entry_string)

    # Remove non-alphabetic tokens, such as punctuation
    words = [word.lower() for word in tokens if word.isalpha()]

    # Filter out stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]

    # pretrained_embeddings_path = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    pretrained_embeddings_path = "GoogleNews-vectors-negative300.bin.gz"

    # Load word2vec model (trained on an enormous Google corpus)
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)
    
    # Filter the list of vectors to include only those that Word2Vec has a vector for
    vector_list = [model[word] for word in words if word in model.vocab]

    # Create a list of the words corresponding to these vectors
    words_filtered = [word for word in words if word in model.vocab]

    # Zip the words together with their vector representations
    word_vec_zip = zip(words_filtered, vector_list)

    # Cast to a dict so we can turn it into a DataFrame
    word_vec_dict = dict(word_vec_zip)
    word2vec_df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    
    return word2vec_df
    

def word2vec_df(model):
    words = list(model.wv.vocab)
    
    # Filter the list of vectors to include only those that Word2Vec has a vector for
    vector_list = [model[word] for word in words if word in model.wv.vocab]

    # Create a list of the words corresponding to these vectors
    words_filtered = [word for word in words if word in model.wv.vocab]

    # Zip the words together with their vector representations
    word_vec_zip = zip(words_filtered, vector_list)

    # Cast to a dict so we can turn it into a DataFrame
    word_vec_dict = dict(word_vec_zip)
    word2vec_df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
#     word2vec_df.head()
    return word2vec_df

def plot(word2vec_df):
    # Initialize t-SNE
    tsne = TSNE(n_components = 2, init = 'random', random_state = 10, perplexity = 100)

    # Use only 400 rows to shorten processing time
    tsne_df = tsne.fit_transform(word2vec_df[:100])

    # dir(sns)
    sns.set()
    # Initialize figure
    fig, ax = plt.subplots(figsize = (11.7, 8.27))
    # sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha = 0.5)
    sns.regplot(tsne_df[:, 0], tsne_df[:, 1], fit_reg=False)

    # Import adjustText, initialize list of texts
#     from adjustText import adjust_text
    # Be sure to import it using the camelcase adjustText, 
    # and please note that adjustText is currently not compatible with matplotlib 3.0 or higher.

    texts = []
    words_to_plot = list(np.arange(0, 100, 5))
    # words_to_plot = list(np.arange(0, 500, 10))

    # Append words to list
    for word in words_to_plot:
        texts.append(plt.text(tsne_df[word, 0], tsne_df[word, 1], word2vec_df.index[word], fontsize = 14))

    # Plot text using adjust_text (because overlapping text is hard to read)
    adjust_text(texts, force_points = 0.4, force_text = 0.4, 
                expand_points = (2,1), expand_text = (1,2),
                arrowprops = dict(arrowstyle = "-", color = 'black', lw = 0.5))

    plt.show()

def plot_embedding(df, columnName, untrained=True, word_embedding_size=300, train_algo=0):
    if untrained:
        model = create_model(df, columnName, word_embedding_size, train_algo)
        word2vec_data = word2vec_df(model)
    else:
        word2vec_data = load_model(df, columnName)
        
    plot(word2vec_data)
#     return (word2vec_data)
    
def word2vec_pretrained(og_df, columnName, tdif):
    df = og_df.copy()
    # Grab all entries
    all_entry = df[columnName]

    # Create a list of strings, one for each entry
    entry_list = [entry for entry in all_entry]

#     # Collapse the list of strings into a single long string for processing
#     big_entry_string = ' '.join(entry_list)

#     # Tokenize the string into words
#     tokens = word_tokenize(big_entry_string)

#     # Remove non-alphabetic tokens, such as punctuation
#     words = [word.lower() for word in tokens if word.isalpha()]

#     # Filter out stopwords
#     stop_words = set(stopwords.words('english'))
#     words = [word for word in words if not word in stop_words]

    # pretrained_embeddings_path = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    pretrained_embeddings_path = "GoogleNews-vectors-negative300.bin.gz"

    # Load word2vec model (trained on an enormous Google corpus)
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)
    
    # adding responses into corpus
    corpus = [preprocess_pretrained(entry) for entry in entry_list]

    # Remove docs that don't include any words in W2V's vocab
    corpus, experience_list = filter_docs_pretrained(corpus, entry_list, lambda doc: has_vector_representation_pretrained(model, doc))

    # Filter out any empty docs
    corpus, experience_list = filter_docs_pretrained(corpus, entry_list, lambda doc: (len(doc) != 0))
    x = []
    # tdif or no tdif 
    if tdif:
        tfid_corpus = []
        for l in corpus:
            tfid_corpus.append(' '.join(l))

        vectorizer = TfidfVectorizer()
        vecs = vectorizer.fit_transform(tfid_corpus)
        
        tfidf_dict = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        
        for doc in corpus: # append the vector for each document
            x.append(document_vector_tdif_pretrained(model, doc, tfidf_dict))
    else:
        for doc in corpus: # append the vector for each document
            x.append(document_vector_pretrained(model, doc))

    X = np.array(x) # list to array
    
    df_w_vectors = pd.DataFrame(x)

    df_w_vectors[columnName] = entry_list
    # Use pd.concat to match original experiences with their vectors
    main_w_vectors = pd.concat((df_w_vectors, df), axis=1)

    # Get rid of vectors that couldn't be matched with the main_df
    main_w_vectors.dropna(axis=0, inplace=True)

    return main_w_vectors

# preprocessing functions to keep experience response intact
# https://github.com/sdimi/average-word2vec/blob/master/notebook.ipynb

def document_vector_tdif_pretrained(model, doc, tfidf_dict):
    """
    return average of all vector representation within text blurb, with weighted words using tdif 
    """
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in model.vocab]
    weighted = []
    for word in doc:
        if word in tfidf_dict:
            weighted.append(tfidf_dict[word] * model[word])
        else:
            weighted.append(model[word])
    return np.mean(weighted, axis = 0)


def document_vector_pretrained(model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in model.vocab]
    return np.mean(model[doc], axis=0) # average vectors 


# Our earlier preprocessing was done when we were dealing only with word vectors
# Here, we need each document to remain a document 
def preprocess_pretrained(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] 
    return doc

# Function that will help us drop documents that have no word vectors in word2vec
def has_vector_representation_pretrained(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)

# Filter out documents
def filter_docs_pretrained(corpus, texts, condition_on_doc):
    """
    Filter corpus and texts given the function condition_on_doc which takes a doc. The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts)