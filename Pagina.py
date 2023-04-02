# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:04:24 2022

@author: Sergio
"""

import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image


from urllib.error import URLError

import sys
sys.path.append('C:/Users/Sergio/Documents/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master')

logo = 'https://ellisalicante.org/assets/xprize/images/logo_oscuro.png'
carpeta = 'C:/Users/Sergio/Documents/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master/'

st.set_page_config(layout = 'wide')

@st.cache
def get_UN_data():
    paises = pd.read_csv(carpeta + "countries_regions.csv")
    return paises.set_index("CountryName")

def get_data_rule(DATA):
    x1 = DATA[0]
    date1 = DATA.index[0]
    xs=[x1]
    dates=[date1]
    for i in range(1, len(data)):
        c = DATA[i]
        if c!= x1:
            x1 = c
            xs.append(x1)
            dates.append(DATA.index[i])
    dates.append(DATA.index[-1])        
    return xs,dates

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
    
    
    st.markdown('# Visualizations of applied NPIs')
    
    with st.expander('Confirmed cases of Covid-19'):
    
        cols = st.columns((1,1,5))
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
            start_date = pd.to_datetime(start_date,format = '%Y-%m-%d')
            
            choose = st.radio("",("Confirmed Cases","Confirmed Deaths"))
        
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
        
        data = data[(data.Date >= start_date)&(data.Date <= end_date)]
        data = data.set_index("Date")
        
        with cols[2]:
            if choose == "Confirmed Cases":
                st.line_chart(data.ConfirmedCases.diff().fillna(0))
            else:
                st.line_chart(data.ConfirmedDeaths.diff().fillna(0))
                
        cols = st.columns((2,5))
        
        with cols[0]:
            rules = ["C1M_School closing","C2M_Workplace closing","C3M_Cancel public events",
                     "C4M_Restrictions on gatherings","C5M_Close public transport",
                     "C6M_Stay at home requirements","C7M_Restrictions on internal movement",
                     "C8EV_International travel controls","H1_Public information campaigns",
                     "H2_Testing policy","H3_Contact tracing","H6M_Facial Coverings"]
            rule = st.multiselect(
                "Choose rule", rules,"C1M_School closing"
            )
            
            value_max = [3,3,2,4,2,3,2,4,2,3,2,4]
            
        with cols[1]:
            if len(rule)!=0:
                for j in range(len(rule)):
                    [xs,dates] = get_data_rule(data[rule[j]].fillna(0))
                    dataf = []
                    for k in range(value_max[j]):
                        dataf.append({rule[j]:str(k),"start":dates[0],"end":dates[0]})
                    for i in range(len(xs)):
                        dataf.append({rule[j]:str(int(xs[i])),"start":dates[i],"end":dates[i+1]})
                                
                    data2 = pd.DataFrame(dataf)      

                    graf = alt.Chart(data2).mark_bar().encode(
                        x=alt.X('start',axis=alt.Axis(title='Date', labelAngle=-45, format = ("%b %Y"))),
                        x2='end',
                        y=rule[j],
                        color = alt.Color(rule[j],legend = None)
                    ).properties(width = 800)
                        
                    st.altair_chart(graf)
    
    # with st.expander('Predict cases of Covid-19'):
    #     cols = st.columns((1,1,3))
    #     paises = get_UN_data()
    #     data_pred = pd.read_csv(carpeta + "predictions/robojudge_test.csv")
    #     data_pred_lineal = pd.read_csv(carpeta + "covid_xprize/examples/predictors/linear/predictions/robojudge_test.csv")
    #     with cols[0]:
    #         country_pred = st.selectbox(
    #             "Choose countrys",list(paises.index.unique())
    #         )
            
    #         data_pred = data_pred[data_pred.CountryName == country_pred]
    #         data_pred['Date'] = pd.to_datetime(data_pred['Date'], format = '%Y-%m-%d')
    #         data_pred_lineal = data_pred[data_pred_lineal.CountryName == country_pred]
    #         data_pred_lineal['Date'] = pd.to_datetime(data_pred_lineal['Date'], format = '%Y-%m-%d')
            
    #         today = min(data_pred.Date)
    #         start_date = st.date_input('Start date', today)
    #         change = st.button("Change the model")
    #         if change:
    #             model = st.selectbox("Choose model",("WIP"))
        
    #     with cols[1]:
    #         reg_pred = list(paises.index).count(country_pred)==1
    #         region = " "
    #         regiones = list(paises[paises.index == country_pred].RegionName.fillna(" "))
    #         region = cols[1].selectbox(
    #              "Choose regions", regiones, disabled = reg_pred
    #         )
    #         tomorrow = max(data_pred.Date)
    #         end_date = st.date_input('End date', tomorrow)
    #         end_date = pd.to_datetime(end_date,format = '%Y-%m-%d')
    #         if start_date > end_date:
    #             st.error('Error: End date must fall after start date.')
  
    #     with cols[2]:
    #         if not reg_pred:
    #             data_pred = data_pred[data_pred.RegionName == region]
    #         data_pred = data_pred.set_index("Date")
    #         data_pred_lineal = data_pred_lineal.set_index("Date")
    #         st.line_chart()  
        
    st.markdown("# Computational epidemiological models")
    cols = st.columns((5,2))
    
    with cols[0]:
        foto1 = Image.open(carpeta + "Foto1.png")
        st.image(foto1)
        
    with cols[1]:
        st.write("We have developed machine learning-based predictive models of the number"
                 "of hospitalizations and intensive care hospitalizations overall and for"
                 "SARS-CoV-2 patients. We have also developed a model to infer the prevalence"
                 "of the disease based on a few of the answers to our citizen survey "
                 "[https://covid19impactsurvey.org](https://covid19impactsurvey.org/)")
        
    st.markdown("# Prescriptor Models")
    cols = st.columns((2,5))
    
    with cols[0]:
        st.write("Our goal in the Prescription phase of the competition is to develop an"
                 "interpretable, data-driven and flexible prescription framework that would"
                 "be usable by non machine-learning experts, such as citizens and policy"
                 "makers in the Valencian Government. Our design principles are therefore"
                 "driven by developing interpretable and transparent models.")
        
        st.write("Given the intervention costs, it automatically generates up to 10"
                 "Pareto-optimal intervention plans. For each plan, it shows the resulting"
                 "number of cases and overall stringency, its position on the Pareto front"
                 "and the activation regime of each of the 12 types of interventions that"
                 "are part of the plan.")
    
    with cols[1]:
        foto2 = Image.open(carpeta + "Foto2.png")
        st.image(foto2)
    
        
except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )
