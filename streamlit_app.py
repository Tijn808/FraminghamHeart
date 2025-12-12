import streamlit as st
import pandas as pd

st.set_page_config(page_title="Predicting Diabetes using the Framingham Heart Study ", layout="wide")
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
    relevant_columns = [
    'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI',
    'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'GLUCOSE',
    'HYPERTEN', 'DIABETES']
    df = data[relevant_columns]
    st.dataframe(df, use_container_width=True, height=300)

#Separating data in features (x) and target (y)
df_relevant = data[relevant_columns]
X = df_relevant.drop('DIABETES', axis=1)
y = df_relevant['DIABETES']

st.markdown('## Train-Test Split')
with st.expander ('# Train-Test Split'):
    st.info('We split the dataset into a training and a testing set, using a 70-30 split.')

# splitting data set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.markdown('## Identifying Problems in the Basic Data')

with st.expander ('Capping'):
    st.info('We capped SYSBP, DIABP, TOTCHOL & BMI at plausible clinical ranges')
    def apply_capping_rules(df):
    # Define clinical ranges for capping
        capping_rules = {
        'SYSBP': {'min': 80, 'max': 260},
        'DIABP': {'min': 40, 'max': 150},
        'TOTCHOL': {'min': 80, 'max': 450},
        'BMI': {'min': 15, 'max': 60}}
        # GLUCOSE is explicitly excluded from capping

        df_copy = df.copy()

        for col, limits in capping_rules.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].clip(lower=limits['min'], upper=limits['max'])
        return df_copy
X_train_capped = apply_capping_rules(X_train)
X_test_capped = apply_capping_rules(X_test)

with st.expander ('# Missing Values'):
    st.info ('As you can see below there were a few variables with missing values:')
    st.dataframe(df.isnull().sum(), use_container_width=True, height=300)

with st.expander ('Imputation'):
    st.info ('...')