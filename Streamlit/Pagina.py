# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:04:24 2022

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

st.set_page_config(layout = 'wide')

@st.cache
def get_UN_data():
    paises = pd.read_csv(carpeta + "countries_regions.csv")
    return paises.set_index("CountryName")

try:
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("")
        
    with col2:
        st.image(logo)
    
    with col3:
       st.write("")
    
    st.markdown('# Who are we?')
    st.write('#### Introducction')
    cols = st.columns((2,1))
    cols[0].write('''We are a team of Spanish scientists who have been working since March 2020 in collaboration with the Valencian Government of Spain on using Data Science to help fight the SARS-CoV-2 pandemic. We have focused on 4 large areas of work: large-scale human mobility modeling via the analysis of aggregated, anonymized data derived from the mobile network infrastructure; computational epidemiological models; predictive models and citizen science by means of a large-scale citizen survey called COVID19impactsurvey which, with over 375,000 answers in Spain and around 150,000 answers from other countries is one of the largest COVID-19 citizen surveys to date. Our work has been awarded two competitive research grants. 
                  \n Since March, we have been developing two types of traditional computational epidemiological models: a metapopulation compartmental SEIR model and an agent-based model. However, for this challenge, we opted for a deep learning-based approach, inspired by the model suggested by the challenge organizers. Such an approach would enable us to build a model within the time frame of the competition with two key properties: be applicable to a large number of regions and be able to automatically learn the impact of the Non-Pharmaceutical Interventions (NPIs) on the transmission rate of the disease. The Pandemic COVID-19 XPRIZE challenge has been a great opportunity for our team to explore new modeling approaches and expand our scope beyond the Valencian region of Spain.''')
    cols[1].video('https://www.youtube.com/watch?v=RZ9wsSGH8U8')
    
    
    st.markdown('# Confitmed cases of Covid-19')
    
    choose = st.radio("View by",("Continent","Country"))
    
    if choose == "Continent":
        st.write('Hola')
    else:
        cols = st.columns((1,1,3))
        paises = get_UN_data()
        data = pd.read_csv(carpeta + "data/OxCGRT_latest.csv")
        with cols[0]:
            country = st.selectbox(
                "Choose country",list(paises.index.unique())
            )
            
            data = data[data.CountryName == country]
            data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
            
            today = min(data.Date)
            start_date = st.date_input('Start date', today)
        
        with cols[1]:
            reg = list(paises.index).count(country)==1
            region = " "
            regiones = list(paises[paises.index == country].RegionName.fillna(" "))
            region = cols[1].selectbox(
                 "Choose region", regiones, disabled = reg
            )
            tomorrow = max(data.Date)
            end_date = st.date_input('End date', tomorrow)
            end_date = pd.to_datetime(end_date,format = '%Y-%m-%d')
            if start_date > end_date:
                st.error('Error: End date must fall after start date.')
        
        data = data.set_index("Date")
        
        with cols[2]:
            st.line_chart(data.ConfirmedCases.diff().fillna(0))
    
    exec(open("predict.py").read())
    
    valencia_output_file = "predictions/robojudge_test.csv"
    $ python carpeta/predict.py -s {start_date_str} -e {end_date_str} -ip {IP_FILE} -o {valencia_output_file}
    
    predictions["V4C"] = valencia_output_file
    
    def get_predictions_from_file(predictor_name, predictions_file, ma_df):
        preds_df = pd.read_csv(predictions_file,
                               parse_dates=['Date'],
                               encoding="ISO-8859-1",
                               error_bad_lines=False)
        preds_df["RegionName"] = preds_df["RegionName"].fillna("")
        preds_df["PredictorName"] = predictor_name
        preds_df["Prediction"] = True
        
        # Append the true number of cases before start date
        ma_df["PredictorName"] = predictor_name
        ma_df["Prediction"] = False
        preds_df = ma_df.append(preds_df, ignore_index=True)
    
        # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
        # np.where usage: if A then B else C
        preds_df["GeoID"] = np.where(preds_df["RegionName"] == '',
                                     preds_df["CountryName"],
                                     preds_df["CountryName"] + ' / ' + preds_df["RegionName"])
        # Sort
        preds_df.sort_values(by=["GeoID","Date"], inplace=True)
        # Compute the 7 days moving average for PredictedDailyNewCases
        preds_df["PredictedDailyNewCases7DMA"] = preds_df.groupby(
            "GeoID")['PredictedDailyNewCases'].rolling(
            WINDOW_SIZE, center=False).mean().reset_index(0, drop=True)
    
        # Put PredictorName first
        preds_df = preds_df[["PredictorName"] + [col for col in preds_df.columns if col != "PredictorName"]]
        return preds_df

    test_predictor_name = "V4C"
    temp_df = get_predictions_from_file(test_predictor_name, predictions[test_predictor_name], ma_df.copy())
    temp_df.head(14)
  
   
        
except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )