import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn; print("Scikit-Learn", sklearn.__version__)

#pipe = pickle.load(open('pipe.pkl', 'rb'))
#df = pickle.load(open('df.pkl', 'rb'))
pipe = pd.read_pickle(open('pipe.pkl', 'rb'))
df = pd.read_pickle(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Brand
Company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
Type = st.selectbox('Type', df['TypeName'].unique())

# Ram
Ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
Weight = st.number_input('Weight of the Laptop')

# Touchscreen
Touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
IPS = st.selectbox('IPS', ['No', 'Yes'])

# Screen Size
Screen_size = st.number_input('Screen Size')

# Resolution
Resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU
CPU = st.selectbox('CPU', df['Cpu Brand'].unique())

HDD = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

SSD = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

GPU = st.selectbox('GPU', df['Gpu brand'].unique())

OS = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # query
    PPI = None
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0

    if IPS == 'Yes':
        IPS = 1
    else:
        IPS = 0

    x_res = int(Resolution.split('x')[0])
    y_res = int(Resolution.split('x')[1])
    PPI = ((x_res**2) + (y_res**2))**0.5/Screen_size

    Query = np.array([Company, Type, Ram, Weight, Touchscreen, IPS, PPI, CPU, HDD, SSD, GPU, OS])

    Query = Query.reshape(1, 12)
    st.title(np.exp(pipe.predict(Query)))


