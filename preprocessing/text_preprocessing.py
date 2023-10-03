from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)
import nltk

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm


def preprocessDataset(train_text):

    train_text = str(train_text)
    tokenized_train_set = text_to_word_sequence(
        train_text, filters="[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+", lower=True, split=" ")

    stop_words = set(stopwords.words('russian'))
    stopwordremove = [i for i in tokenized_train_set if not i in stop_words]

    stopwordremove_text = ' '.join(stopwordremove)
    numberremove_text = ''.join(
        c for c in stopwordremove_text if not c.isdigit())

    outp = []

    doc = Doc(numberremove_text)
    doc.segment(segmenter)
    doc.tag_morph(tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        outp.append(token.lemma)

    lem_text = ' '.join(outp)

    return lem_text


import re


def clean_text(input_text):
    clean_text = re.sub('<[^<]+?>', '', input_text)

    clean_text = re.sub(r'http\S+', '', clean_text)

    clean_text = re.sub('@[^\s]+', '', clean_text)

    clean_text = re.sub('#[^\s]+', '', clean_text)

    clean_text = re.sub(' +', ' ', clean_text)

    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"
                      u"\U0001F300-\U0001F5FF"
                      u"\U0001F680-\U0001F6FF"
                      u"\U0001F1E0-\U0001F1FF"
                      u"\U00002500-\U00002BEF"
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
                      u"\ufe0f"
                      u"\u3030"
                      "]+", re.UNICODE)
    clean_text = re.sub(emoj, '', clean_text).replace('\n', '')

    return clean_text.strip()
