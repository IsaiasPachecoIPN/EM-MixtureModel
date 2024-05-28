import spacy
import es_core_news_sm

#Preprocessing functions
def remove_stopwords(text, stop_words_path):

    """
    Function to remove the stopwords from a text
    @param text:    The text to remove the stopwords
    @param stop_words_path: path to the file with the stopwords
    @return:        The text without the stopwords
    """

    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stop_words = f.read().splitlines() #TODO checar estp
        # print(f'Stop words: {stop_words}')

    return ' '.join([word for word in text.split() if word not in stop_words])


def lower_text(text):
    """
    Function to lower the text
    @param text:    The text to lower
    @return:        The text lowered
    """
    return text.lower()

def remove_punctuation(text):
    """
    Function to remove the punctuation from a text
    @param text:    The text to remove the punctuation
    @return:        The text without the punctuation
    """
    
    puntuation_signs = ['.',',',';',':','!','¡','¿','?','(',')','[',']','{','}','"','/','\\','|','<','>','@','#','$','%','^','&','*','_','+','-','=','~','`']

    for sign in puntuation_signs:
        text = text.replace(sign, '')

    return text      

def remove_numerical_values(text):
    """
    Function to remove the numerical values from a text
    @param text:    The text to remove the numerical values
    @return:        The text without the numerical values
    """
    return ''.join([i for i in text if not i.isdigit()])  

def lemmatize_text(text, verbose=False):
    """
    Function to lemmatize the text
    @param text:    The text to lemmatize
    @return:        The lemmatized text
    """

    nlp = spacy.load('es_core_news_sm')
    nlp = es_core_news_sm.load()

    doc = nlp(text)
    lemmatized_text = ' '.join([word.lemma_ for word in doc])

    if verbose:
        print(f'Lemmatized text: {lemmatized_text}')
    
    return lemmatized_text