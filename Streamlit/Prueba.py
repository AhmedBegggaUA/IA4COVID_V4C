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
    paises = get_UN_data()
    country = st.selectbox(
        "Choose country",list(paises.index.unique())
    )
    
    if list(paises.index).count(country)!=1:
        regiones = list(paises.loc[country].RegionName.fillna("None"))
        region = st.selectbox(
            "Choose region", regiones
        )
        
   

    data = paises.loc[country]
    data /= 1000000.0
    st.write("### Predicted cases of Covid 19", data.sort_index())

    data = data.T.reset_index()
    data = pd.melt(data, id_vars=["index"]).rename(
        columns={"index": "data", "value": "Predicted cases"}
    )
    chart = (
        alt.Chart(data)
        .mark_line(opacity=0.3)
        .encode(
            x="year:T",
            y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
            color="Region:N",
        )
    )
    st.altair_chart(chart, use_container_width=True)
except URLError as e:
    st.error(
        """
    
        **This demo requires internet access.**

        Connection error: %s
    """

        % e.reason
    )
