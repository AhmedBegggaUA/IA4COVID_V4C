# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:18:00 2022

@author: Sergio
"""

import streamlit as st
import pandas as pd
import altair as alt

import json
import logging
import numpy as np

from urllib.error import URLError

import sys
sys.path.append('C:/Users/Sergio/Desktop/Trabajo/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master')

logo = 'https://ellisalicante.org/assets/xprize/images/logo_oscuro.png'
carpeta = 'C:/Users/Sergio/Desktop/Trabajo/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master/'

def interactive_galaxies(df,data):
    
    data = data[data.CountryName == country]
    data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
    today = min(data.Date)
    tomorrow = max(data.Date)
    
    questions = {
        'country': df.index.unique(),
        'region' :list(df[df.index == country].RegionName.fillna(" "))
    }
    # could make merging yes/no

    # st.sidebar.markdown('# Show posteriors')
    # show_posteriors = st.sidebar.selectbox('Posteriors for which question?', ['none'] + list(questions.keys()), format_func=lambda x: x.replace('-', ' ').capitalize())

    st.sidebar.image(logo)

    st.sidebar.markdown('# Choose a country')
    # st.sidebar.markdown('---')
    for question, answers in questions.items():
        selected_answer = st.sidebar.selectbox(question.replace('-', ' ').capitalize(),answers, format_func=lambda x: x.replace('-',' ').capitalize(), key=question+'_select')

    start_date = st.sidebar.date_input('Start date', today)
    start_date = pd.to_datetime(start_date,format = '%Y-%m-%d')
    end_date = st.sidebar.date_input('End date', tomorrow)
    end_date = pd.to_datetime(end_date,format = '%Y-%m-%d')

    if st.sidebar.button('More settings'):
        questions2 ={
            'Models':['Predet'],
            'Training':['Predet'],
            }
        for question2, answers in questions2.items():
            selected_answer = st.sidebar.selectbox(question2.replace('-', ' ').capitalize(),answers, format_func=lambda x: x.replace('-',' ').capitalize(), key=question2+'_select')
        
        

@st.cache
def get_UN_data():
    paises = pd.read_csv(carpeta + "countries_regions.csv")
    return paises.set_index("CountryName")

try:
    st.image(logo,width=600)
    cols = st.columns((1,1))
    paises = get_UN_data()
    country = cols[0].selectbox(
        "Choose country",list(paises.index.unique())
    )
    reg = list(paises.index).count(country)==1
    region = " "
    regiones = list(paises[paises.index == country].RegionName.fillna(" "))
    region = cols[1].selectbox(
         "Choose region", regiones, disabled = reg
    )
        
    data = pd.read_csv(carpeta + "data/OxCGRT_latest.csv")
    interactive_galaxies(paises,data)
    data = data[data.CountryName == country]
    data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
    
    col1, col2, col3 = st.columns([6,6,6])

    with col1:
        st.write("")
    
    with col2:
        st.button('Predict')
    
    with col3:
        st.write("")
    
    if not reg:
        data = data[data.RegionName == region]
    
    data = data.set_index("Date")
    

        
  
   
        
except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )