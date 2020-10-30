#/home/jaideep/Desktop/folder/ML DS/datasets/MultiClass-Text-Classification-master/Consumer_Complaints.csv
import csv
import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import cleantext
from txt_to_csv import csv_dataset

st.title("AI Email Classifier {IITB Techfest 2020}")
dataset = st.sidebar.text_input('Enter Dataset File Path:')
#dataset = csv_dataset(dataset)
try:
    with open(dataset) as input:
        input.read()
except FileNotFoundError:
    st.sidebar.error('File not found.')

#st.write(dataset)
if dataset:
    df = pd.read_csv(dataset)
    st.write("### Initial Dataset")
    st.write(df.head())
    st.bar_chart(df['Product'].value_counts(), height = 400)
    # TEXT PREPROCESSING
    if st.sidebar.button(label="Preprocess The Text"):
        df_to_use = cleantext(df)
        st.write(df_to_use.head())
        st.write(df_to_use.shape)

    vectorizer = st.sidebar.selectbox("Select Vectorizer", ("Word2Vec", "TF-IDF", "BERT"))
    classifier = st.sidebar.selectbox("Select Classifier Algorithm", ("LinearSVC", "GridSearchCV", "Logistic Regression"))
    #st.balloons()
