from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import re
lemmatizer = WordNetLemmatizer()
def cleantext(df):
    df_to_use = df[['Product','Consumer Complaint']]
    df_to_use.rename(columns={"Consumer Complaint": "Consumer_Complaint"}, inplace = True)
    # select = st.sidebar.selectbox('Replace Null Values With',('Remove Rows','Mode'))
    # if select == 'Remove Rows':
    #     rem_null_values(df_to_use)
    # if select == 'Mode':
    #     modify_null_values(df_to_use)
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
    sen = sen.lower()
    return sen

# def rem_stopwords():
#     return 

# def rem_null_values(df_to_use):
#     df_to_use = df_to_use.dropna(axis=0, inplace = True)
#     return df_to_use

# def modify_null_values(df_to_use):   #replace with mode
#     df_to_use = df_to_use['Consmer_Complaint'].fillna(df_to_use['Consumer_Complaint'].mode()[0], inplace=True)
#     return df_to_use