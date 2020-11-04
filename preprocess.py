from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import re
lemmatizer = WordNetLemmatizer()
def cleantext(df):
    df_to_use = df[['Product','Consumer Complaint']]
    df_to_use.rename(columns={"Consumer Complaint": "Consumer_Complaint"}, inplace = True)
    df_to_use.dropna(axis=0, inplace = True)
    df_to_use['Consumer_Complaint'] = df_to_use.Consumer_Complaint.apply(lemmatize_text)
    df_to_use['Consumer_Complaint'] = df_to_use.Consumer_Complaint.apply(rem_splchars)
    #remove stopwords
    #reduce dimentionality
    return df_to_use

def lemmatize_text(sen):
    return [lemmatizer.lemmatize(word) for word in word_tokenize(sen)]

def rem_splchars(sen):
    sen = re.sub(r'\W', ' ', str(sen))
    sen = re.sub(r'\^[a-zA-Z]\s+', ' ', str(sen))
    sen = re.sub(r'\s+', ' ', sen, flags=re.I)
    sen = re.sub(r'xxxx','',str(sen))
    return sen.lower()

# def rem_stopwords():
#     return 
