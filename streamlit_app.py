import streamlit as st
import pandas as pd
# project name

# research question
st.markdown('## Research Question')
with st.expander('# Research Question'):
    st.markdown ('## Initial Research Question')
    st.info('Can we predict the onset of diabetes in the Framingham Heart Study population using baseline demographic, lifestyle, and clinical variables?')
    st.divider()
    st.write ('An analysis of existing studies showed that diabetes has not been extensively investigated in previous research. Recognizing this gap, we aimed to contribute to a deeper understanding of this condition.')
    st.markdown ('## Redefined Research Question')
    st.info('Can we identify individuals currently positive for or at high risk of diabetes within the Framingham Heart Study population, using readily available baseline demographic, lifestyle, and clinical variables such as age, sex, BMI, blood pressure, cholesterol, glucose, and smoking status?')

#  column selection
st.markdown ('## Column Selection')
with st.expander ('Selected Columns'):
    st.write('Selected columns are: age, sex, totchol, sysbp, diabp, cursmoke, cigpday, BMI, bpmeds, prevchd, prevap, prevmi, prevstrk, prevhyp, glucose, hyperten & diabetes')
    #explain why these are chosen variables
    st.divider()
    data = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')
    data.head()
    data_raw = data.copy(deep=True) #so the data keeps it original state
    selected_columns = [
    'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI',
    'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'GLUCOSE',
    'HYPERTEN', 'DIABETES']
    df = data[selected_columns]
    st.dataframe(df, use_container_width=True, height=300)

st.markdown('## Train-Test Split')
with st.expander ('# Train-Test Split'):
    st.info('We .....')

st.markdown('## Identifying Problems in the Basic Data')
with st.expander ('# Missing Values'):
    st.info ('Handling missing values')
    st.dataframe(df.isnull().sum(), use_container_width=True, height=300)

with st.expander ('Imputation'):
    st.info ('...')