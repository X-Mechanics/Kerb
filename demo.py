import streamlit as st
from kerb import kerb

st.sidebar.header('Datasets')
query = st.sidebar.text_area("Query", "", height=250)

if st.sidebar.button("Search"):
    parsed = query.split(' ')
    results = kerb(query)