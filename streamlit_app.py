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

st.markdown('## Identifying Problems in the Data')

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
    st.info ('We used several types of imputation depending on the variable.')
    st.write ('For BMI & TOTCHOL: median imputation')
    st.write ('For CIGPDAY: if CURSMOKE is 0, 0 imputation for CIGPDAY, if CURSMOKE is 1, use median imputation for CIGPDAY')
    st.write('For BPMEDS: 0 imputation')
    st.write('For GLUCOSE: GLUCOSE_missing indicator column with missing values imputed using the 70th percentile of GLUCOSE from X_train_capped')

#BMI imputation
median_bmi = X_train_capped['BMI'].median()
X_train_capped['BMI'].fillna(median_bmi, inplace=True)
X_test_capped['BMI'].fillna(median_bmi, inplace=True)

#TOTCHOL imputation
median_totchol = X_train_capped['TOTCHOL'].median()
X_train_capped['TOTCHOL'] = X_train_capped['TOTCHOL'].fillna(median_totchol)
X_test_capped['TOTCHOL'] = X_test_capped['TOTCHOL'].fillna(median_totchol)

#CIGPDAY imputation
X_train_capped.loc[(X_train_capped['CURSMOKE'] == 0) & (X_train_capped['CIGPDAY'].isnull()), 'CIGPDAY'] = 0
X_test_capped.loc[(X_test_capped['CURSMOKE'] == 0) & (X_test_capped['CIGPDAY'].isnull()), 'CIGPDAY'] = 0
median_cigpday_smoker = X_train_capped[X_train_capped['CURSMOKE'] == 1]['CIGPDAY'].median()
X_train_capped['CIGPDAY'] = X_train_capped['CIGPDAY'].fillna(median_cigpday_smoker)
X_test_capped['CIGPDAY'] = X_test_capped['CIGPDAY'].fillna(median_cigpday_smoker)

#BPMEDS imputatio
X_train_capped['BPMEDS'] = X_train_capped['BPMEDS'].fillna(0)
X_test_capped['BPMEDS'] = X_test_capped['BPMEDS'].fillna(0)

#GLUCOSE imputation
X_train_capped['GLUCOSE_missing'] = X_train_capped['GLUCOSE'].isnull().astype(int)
X_test_capped['GLUCOSE_missing'] = X_test_capped['GLUCOSE'].isnull().astype(int)
percentile_70_glucose = X_train_capped['GLUCOSE'].quantile(0.80)
X_train_capped['GLUCOSE'] = X_train_capped['GLUCOSE'].fillna(percentile_70_glucose)
X_test_capped['GLUCOSE'] = X_test_capped['GLUCOSE'].fillna(percentile_70_glucose)

with st.expander('Standardization'):
    st.info('We standardized the data using StandardScaler.')

# Standardization
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
numerical_cols_for_scaling = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'GLUCOSE']
binary_cols_for_passthrough = ['SEX', 'CURSMOKE', 'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'HYPERTEN', 'GLUCOSE_missing']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols_for_scaling),
        ('bin', 'passthrough', binary_cols_for_passthrough)
    ])
X_train_processed_array = preprocessor.fit_transform(X_train_capped)
X_test_processed_array = preprocessor.transform(X_test_capped)
processed_feature_names = numerical_cols_for_scaling + binary_cols_for_passthrough
X_train_processed = pd.DataFrame(X_train_processed_array, columns=processed_feature_names, index=X_train_capped.index)
X_test_processed = pd.DataFrame(X_test_processed_array, columns=processed_feature_names, index=X_test_capped.index)

import matplotlib.pyplot as plt
import seaborn as sns
st.markdown('## Data Visualization')
with st.expander ('Data Distributions'):
    with st.expander ('Categorical Variables'):
        st.markdown ('#### Categorical Variables')
        categorical_cols = ['SEX', 'CURSMOKE', 'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'HYPERTEN', 'GLUCOSE_missing']
        categorical_cols_present = [col for col in categorical_cols if col in X_train_processed.columns]
        if not categorical_cols_present:
            print("No categorical columns found for plotting.")
        else:
            selected_col = st.selectbox('Select a categorical column to visualize:', categorical_cols_present)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=X_train_processed[selected_col], ax=ax)
            ax.set_title(f"Distribution of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    with st.expander ('Numerical Variables'):
        st.markdown ('#### Numerical Variables')
        numerical_cols = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'GLUCOSE']
        numerical_cols_present = [col for col in numerical_cols if col in X_train_processed.columns]
        if not numerical_cols_present:
            print("No numerical columns found for plotting.")
        else:
            selected_col = st.selectbox('Select a numerical column to visualize:', numerical_cols_present)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(X_train_processed[selected_col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
    
            #shows the big imbalance in data:
    with st.expander ('Diabetes Distribution'):
        st.markdown('#### Diabetes Distribution')
        if isinstance(y_train, pd.DataFrame):
            y_train_series = y_train.iloc[:, 0]
        else:
            y_train_series = y_train
        fig, ax = plt.subplots(figsize=(6, 4))  # << assign to fig!
        sns.countplot(x=y_train_series, ax=ax)
        ax.set_title('Distribution of DIABETES in Training Data')
        ax.set_xlabel('DIABETES (0: No Diabetes, 1: Diabetes)')
        ax.set_ylabel('Frequency')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Diabetes', 'Diabetes'])
        st.pyplot(fig)

with st.expander ('Boxplots of after capping'):
    st.markdown('#### Boxplots after capping')
    capped_columns = ['SYSBP', 'DIABP', 'TOTCHOL', 'BMI']
    n_cols = 2 
    n_rows = len(capped_columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    fig.suptitle('Boxplots of Numerical Features Before and After Capping', y=1.02, fontsize=16)
    for i, col in enumerate(capped_columns):
        sns.boxplot(y=X_train[col], ax=axes[i, 0])
        axes[i, 0].set_title(f'{col} - Before Capping')
        axes[i, 0].set_ylabel(col)
        sns.boxplot(y=X_train_capped[col], ax=axes[i, 1])
        axes[i, 1].set_title(f'{col} - After Capping')
        axes[i, 1].set_ylabel(col)
    plt.tight_layout() 
    st.pyplot(fig)

       #MAKE IT DROPCHOICE