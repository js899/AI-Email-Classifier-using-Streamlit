#/home/jaideep/Desktop/folder/ML DS/datasets/MultiClass-Text-Classification-master/Consumer_Complaints.csv
import csv
import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import cleantext
from txt_to_csv import csv_dataset
from tfidf_lsvc import tfidf_lsvc_train

st.title("AI Email Classifier {IITB Techfest 2020}")
dataset = st.sidebar.text_input('Enter Dataset Path:')
#dataset = csv_dataset(dataset)
# try:
#     with open(dataset) as input:
#         input.read()
# except FileNotFoundError:
#     st.sidebar.error('File not found.')
#dataset = csv_dataset(dataset)

df = pd.read_csv(dataset)
df.drop_duplicates(subset=None, inplace=True)
df.to_csv(os.getcwd()+"/datasets/data.csv", index=False)
st.write("### Initial Dataset")
st.write(df.head())
st.write(df.shape)
st.write('#### Category Count Plot')
st.bar_chart(df['Product'].value_counts(), height = 400)
	
cols = []
cols = st.multiselect("Select Columns to Preprocess", df.columns.tolist(), default=cols)
df_to_use = df[cols]
st.write(df_to_use.head())
if st.sidebar.button(label="Preprocess Text and Quick Test"):
	df_to_use = cleantext(df_to_use)
st.write(df_to_use.head())
st.write(df_to_use.shape)
vectorizer = st.sidebar.selectbox("Select Vectorizer", ("TF-IDF", "Word2Vec"))
classifier = st.sidebar.selectbox("Select Classifier Algorithm", ("LinearSVC", "MultinomialNB"))
if (vectorizer=="TF-IDF" and classifier=="LinearSVC"):
	accuracy = tfidf_lsvc_train(df_to_use)
	st.write("#### Accuracy of TF-IDF with LSVC: ", accuracy)
