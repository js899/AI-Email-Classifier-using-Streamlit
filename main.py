import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("AI Email Classifier {IITB Techfest 2020}")
dataset = st.sidebar.text_input('Enter Dataset File Path:')
try:
    with open(dataset) as input:
        input.read()
except FileNotFoundError:
    st.sidebar.error('File not found.')

st.sidebar.write('You selected `%s`' % dataset)
#st.write(dataset_name)
classifier_name = st.sidebar.selectbox("Select Classifier", ("SVM SVC", "GridSearchCV", "LinearSVC"))
df = pd.read_csv(dataset)
st.write("### Initial Dataset")
st.write(df.head())
