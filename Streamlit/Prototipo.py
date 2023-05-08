
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:26:19 2022

@author: Sergio
"""

import streamlit as st
import pandas as pd
import altair as alt

from urllib.error import URLError

import sys
sys.path.append('C:/Users/Sergio/Desktop/Trabajo/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master')


carpeta = 'C:/Users/Sergio/Desktop/Trabajo/valencia-ia4covid-xprize-master/valencia-ia4covid-xprize-master/'

@st.cache
def get_UN_data():
    paises = pd.read_csv(carpeta + "countries_regions.csv")
    return paises.set_index("CountryName")

try:
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
    data = data[data.CountryName == country]
    data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
    
    today = min(data.Date)
    tomorrow = max(data.Date)
    cols = st.columns(2)
    start_date = cols[0].date_input('Start date', today)
    start_date = pd.to_datetime(start_date,format = '%Y-%m-%d')
    end_date = cols[1].date_input('End date', tomorrow)
    end_date = pd.to_datetime(end_date,format = '%Y-%m-%d')
    if start_date < end_date:
        st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    else:
        st.error('Error: End date must fall after start date.')

    data = data[(data.Date >= start_date)&(data.Date <= end_date)]
    
    data["RegionName"] = data["RegionName"].fillna(" ")
    
    if not reg:
        data = data[data.RegionName == region]
    
    data = data.set_index("Date")

    st.line_chart(data.ConfirmedCases.diff().fillna(0))
    st.line_chart(data.ConfirmedDeaths.diff().fillna(0))
        

    source = pd.DataFrame([
        {"task": "A", "start": 1, "end": 3},
        {"task": "B", "start": 3, "end": 8},
        {"task": "C", "start": 8, "end": 10}
    ])
    
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
    
    rules = ["C1_School closing","C2_Workplace closing","C3_Cancel public events",
             "C4_Restrictions on gatherings","C5_Close public transport",
             "C6_Stay at home requirements","C7_Restrictions on internal movement",
             "C8_International travel controls","H1_Public information campaigns",
             "H2_Testing policy","H3_Contact tracing","H6_Facial Coverings"]
    rule = st.multiselect(
        "Choose rule", rules,"C1_School closing"
    )
    
    value_max = [3,3,2,3,2,3,2,3,2,3,2,4]
    
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
            )      
                
            st.altair_chart(graf)
            
    else: 
        st.error('Error: Choose some rule')
        
  
   
        
except URLError as e:
    st.error(
        """
            
        **This demo requires internet access.**

        Connection error: %s
        """

        % e.reason
    )
