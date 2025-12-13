import streamlit as st
import pandas as pd

st.markdown('# Diabetes Risk Prediction using the Framingham Heart Study Dataset')
st.divider()

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
    st.info('The columns we selected are: AGE, SEX, TOTCHOL (total cholesterol), SYSBP (systolic blood pressure (diastolic blood pressure), CURSMOKE (current smoker), CIGPDAY (cigarettes per day), BMI, BPMEDS (blood pressure medication), PREVCHD (history of coronary heart disease), PREVAP (history of angina pectoris), PREVMI (history of myocardial infarction), PREVSTRK (history of stroke), PREVHYP (history of hypertension), GLUCOS (glucose), HYPERTEN (hypertension) & DIABETES')
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
    st.info('We split the dataset into a training and a testing set, using a 70-30 split. We chose this split to ensure a suffienct amount of diabetes in the test set, given the class imbalance in the Framingham Heart Study dataset. Stratified splitting preserves the class distribution in both sets, making our model performance evaluation more reliable.')

# splitting data set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.markdown('## Identifying Problems in the Data')

with st.expander ('Capping'):
    st.info('We capped SYSBP (systolic blood pressure), DIABP (diastolic blood pressure), TOTCHOL (total cholesterol) & BMI at plausible clinical ranges to reduce the influence of extreme outliers. Values that lie above these clinical ranges are often measurement errors or biological implausible, and can distort the model training. By capping these variables we can ensure that our models learn from realistic data while still preserving the underlying patterns in the data.')
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
    st.info ('As you can see below there are a few variables with missing values:')
    st.dataframe(df.isnull().sum(), use_container_width=True, height=300)

with st.expander ('Imputation'):
    st.info ('We used several types of imputation depending on the variable.')
    st.divider()
    st.info ('For BMI & TOTCHOL: median imputation')
    st.write ('We chose median imputation for BMI and TOTCHOL because these variables are continuous with some extreme values. Median imputation provides a central value, so we do not introduce bias in the dataset.')
    st.info ('For CIGPDAY: if CURSMOKE = 0 we imputed zero; if CURSMOKE = 1 we imputed the median CIGPDAY')
    st.write ('We imputed 0 for nonsmokers (CURSMOKE = 0), since they do not smoke. For smokers (CURSMOKE = 1) we imputed the median CIGPDAY value to represent a typical smoking habit.')
    st.info ('For BPMEDS: 0 imputation')
    st.write ('We imputed zero for BPMEDS, because a missing value is more likely to indicate that the individual is not on blood pressure medication, than that it is an unknown error. This approach minimizes the risk of falsely assuming medication use.')
    st.info ('For GLUCOSE: we imputed the 70th percentile of GLUCOSE from X_train_capped')
    st.write ('We calculated tehe 70th percentile of glucose values from the X_train_capped set. We chose the 70th percentile to avoid underestimating glucose levels for patients that might truly have elevated glucose.')

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

with st.expander('Glucose Feature Engineering'):
    st.info('We created a new binary feature "GLUCOSE_missing to help the model remember which glucose values were missing, because the missingness might not be at random. This preserves the predicitve power of our model.')

with st.expander('Standardization'):
    st.info('We standardized the data using StandardScaler to scale each feature around 0 with a standard deviation of 1. This ensures a even influence of all features; it keeps the relative differences between values intact while making all features comparable')

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

with st.expander ('Boxplots after capping'):
    st.markdown('#### Boxplots after capping')
    capped_columns = ['SYSBP', 'DIABP', 'TOTCHOL', 'BMI']
    selected_col = st.selectbox('Select a column to see before & after capping:', capped_columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Boxplot of {selected_col} - Before and After Capping', fontsize=16, y=1.05)
    sns.boxplot(y=X_train[selected_col], ax=axes[0])
    axes[0].set_title(f'{selected_col} - Before Capping')
    axes[0].set_ylabel(selected_col)
    sns.boxplot(y=X_train_capped[selected_col], ax=axes[1])
    axes[1].set_title(f'{selected_col} - After Capping')
    axes[1].set_ylabel(selected_col)
    plt.tight_layout() 
    st.pyplot(fig)

# More standardization?
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

#Logistic Regression
st.markdown('## Model Training')
with st.expander ('Logistic Regression (unweighted vs. weighted)'):
    st.info('We started with training a logistic regression model to predict diabetes. We also trained a weighted logistic regression model to account for the class imbalance in the data.')
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    # unweighted model
    log_reg_model = LogisticRegression(random_state=42, solver='liblinear') 
    log_reg_model.fit(X_train_processed, y_train)
    y_pred_log_reg = log_reg_model.predict(X_test_processed)
    y_proba_log_reg = log_reg_model.predict_proba(X_test_processed)[:, 1]
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    precision_log_reg = precision_score(y_test, y_pred_log_reg)
    recall_log_reg = recall_score(y_test, y_pred_log_reg)
    f1_log_reg = f1_score(y_test, y_pred_log_reg)
    auc_log_reg = roc_auc_score(y_test, y_proba_log_reg)
    # weighted model
    log_reg_weighted_model = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear')
    log_reg_weighted_model.fit(X_train_processed, y_train)
    y_pred_log_reg_weighted = log_reg_weighted_model.predict(X_test_processed)
    y_proba_log_reg_weighted = log_reg_weighted_model.predict_proba(X_test_processed)[:, 1]
    accuracy_log_reg_weighted = accuracy_score(y_test, y_pred_log_reg_weighted)
    precision_log_reg_weighted = precision_score(y_test, y_pred_log_reg_weighted)
    recall_log_reg_weighted = recall_score(y_test, y_pred_log_reg_weighted)
    f1_log_reg_weighted = f1_score(y_test, y_pred_log_reg_weighted)
    auc_log_reg_weighted = roc_auc_score(y_test, y_proba_log_reg_weighted)
    #comparison
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Performance Comparison: Basic vs. Weighted Logistic Regression', fontsize=16)
    cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
    sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Basic LR: Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['No Diabetes', 'Diabetes'])
    axes[0].set_yticklabels(['No Diabetes', 'Diabetes'])
    cm_log_reg_weighted = confusion_matrix(y_test, y_pred_log_reg_weighted)
    sns.heatmap(cm_log_reg_weighted, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Weighted LR: Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_xticklabels(['No Diabetes', 'Diabetes'])
    axes[1].set_yticklabels(['No Diabetes', 'Diabetes'])
    plt.tight_layout()
    st.pyplot(fig)
    #Show metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_pred_log_reg = log_reg_model.predict(X_test_processed)
    y_proba_log_reg = log_reg_model.predict_proba(X_test_processed)[:, 1]
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    precision_log_reg = precision_score(y_test, y_pred_log_reg)
    recall_log_reg = recall_score(y_test, y_pred_log_reg)
    f1_log_reg = f1_score(y_test, y_pred_log_reg)
    auc_log_reg = roc_auc_score(y_test, y_proba_log_reg)
    
    y_pred_log_reg_weighted = log_reg_weighted_model.predict(X_test_processed)
    y_proba_log_reg_weighted = log_reg_weighted_model.predict_proba(X_test_processed)[:, 1]
    accuracy_log_reg_weighted = accuracy_score(y_test, y_pred_log_reg_weighted)
    precision_log_reg_weighted = precision_score(y_test, y_pred_log_reg_weighted)
    recall_log_reg_weighted = recall_score(y_test, y_pred_log_reg_weighted)
    f1_log_reg_weighted = f1_score(y_test, y_pred_log_reg_weighted)
    auc_log_reg_weighted = roc_auc_score(y_test, y_proba_log_reg_weighted)
    results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Basic Logistic Regression': [accuracy_log_reg,
        precision_log_reg,
        recall_log_reg,
        f1_log_reg,
        auc_log_reg],
    'Weighted Logistic Regression': [
        accuracy_log_reg_weighted,
        precision_log_reg_weighted,
        recall_log_reg_weighted,
        f1_log_reg_weighted,
        auc_log_reg_weighted]})
    st.markdown("#### Logistic Regression Comparison Table")
    st.dataframe(results_df.style.format({'Basic Logistic Regression': "{:.4f}",'Weighted Logistic Regression': "{:.4f}",}))

with st.expander ('Optimized Logistic Regression'):
    import numpy as np 
    thresholds = np.arange(0, 1.01, 0.01)
    best_threshold_f1 = 0
    max_f1_score = 0
    for t in thresholds:
        y_pred_t = (y_proba_log_reg >= t).astype(int)
        current_f1 = f1_score(y_test, y_pred_t)
        if current_f1 > max_f1_score:
            max_f1_score = current_f1
            best_threshold_f1 = t
    y_pred_log_reg_optimized = (y_proba_log_reg >= best_threshold_f1).astype(int)
    accuracy_log_reg_optimized = accuracy_score(y_test, y_pred_log_reg_optimized)
    precision_log_reg_optimized = precision_score(y_test, y_pred_log_reg_optimized)
    recall_log_reg_optimized = recall_score(y_test, y_pred_log_reg_optimized)
    f1_log_reg_optimized = f1_score(y_test, y_pred_log_reg_optimized)
    auc_log_reg_optimized = roc_auc_score(y_test, y_proba_log_reg)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_log_reg_optimized)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Optimized LR Confusion Matrix (Threshold={best_threshold_f1:.2f})')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    ax.set_yticklabels(['No Diabetes', 'Diabetes'])
    st.pyplot(fig)
    #metrics
    st.markdown('#### Optimized Logistic Regression Metrics')
    optimized_df = pd.DataFrame({
    'Metric': ['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Optimized Logistic Regression': [
        best_threshold_f1,
        accuracy_log_reg_optimized,
        precision_log_reg_optimized,
        recall_log_reg_optimized,
        f1_log_reg_optimized,
        auc_log_reg_optimized]})
    st.dataframe(optimized_df.style.format({'Optimized Logistic Regression': "{:.4f}"}))

with st.expander ('Decision Tree'):
    from sklearn.tree import DecisionTreeClassifier
    #unweighted
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train_processed, y_train)
    y_pred_decision_tree = decision_tree_model.predict(X_test_processed)
    y_proba_decision_tree = decision_tree_model.predict_proba(X_test_processed)[:, 1]
    accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
    precision_decision_tree = precision_score(y_test, y_pred_decision_tree)
    recall_decision_tree = recall_score(y_test, y_pred_decision_tree)
    f1_decision_tree = f1_score(y_test, y_pred_decision_tree)
    auc_decision_tree = roc_auc_score(y_test, y_proba_decision_tree)
    #weighted
    decision_tree_weighted_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    decision_tree_weighted_model.fit(X_train_processed, y_train)
    y_pred_decision_tree_weighted = decision_tree_weighted_model.predict(X_test_processed)
    y_proba_decision_tree_weighted = decision_tree_weighted_model.predict_proba(X_test_processed)[:, 1]
    accuracy_decision_tree_weighted = accuracy_score(y_test, y_pred_decision_tree_weighted)
    precision_decision_tree_weighted = precision_score(y_test, y_pred_decision_tree_weighted)
    recall_decision_tree_weighted = recall_score(y_test, y_pred_decision_tree_weighted)
    f1_decision_tree_weighted = f1_score(y_test, y_pred_decision_tree_weighted)
    auc_decision_tree_weighted = roc_auc_score(y_test, y_proba_decision_tree_weighted)
    #confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)
    sns.heatmap(cm_decision_tree, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Basic DT: Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['No Diabetes', 'Diabetes'])
    axes[0].set_yticklabels(['No Diabetes', 'Diabetes'])
    cm_decision_tree_weighted = confusion_matrix(y_test, y_pred_decision_tree_weighted)
    sns.heatmap(cm_decision_tree_weighted, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Weighted DT: Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_xticklabels(['No Diabetes', 'Diabetes'])
    axes[1].set_yticklabels(['No Diabetes', 'Diabetes'])
    plt.tight_layout()
    st.pyplot(fig)
    #Metrics
    decisiontree_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Basic Decision Tree': [accuracy_decision_tree,
        precision_decision_tree,
        recall_decision_tree,
        f1_decision_tree,
        auc_decision_tree],
    'Weighted Decision Tree': [
        accuracy_decision_tree_weighted,
        precision_decision_tree_weighted,
        recall_decision_tree_weighted,
        f1_decision_tree_weighted,
        auc_decision_tree_weighted]})
    st.markdown('#### Decision Tree Comparison Table')
    st.dataframe(decisiontree_df.style.format({'Model Performance Comparison Table': "{:.4f}"}))

with st.expander('Optimized Decision Tree'):
    thresholds = np.arange(0, 1.01, 0.01)   
    best_threshold_f1_dt = 0
    max_f1_score_dt = 0
    best_threshold_gmean_dt = 0
    max_gmean_score_dt = 0
    for threshold in thresholds:
        y_pred_thresholded_dt = (y_proba_decision_tree >= threshold).astype(int)
        current_f1_score_dt = f1_score(y_test, y_pred_thresholded_dt)
        sensitivity_dt = recall_score(y_test, y_pred_thresholded_dt, pos_label=1)
        specificity_dt = recall_score(y_test, y_pred_thresholded_dt, pos_label=0)
        current_gmean_score_dt = np.sqrt(sensitivity_dt * specificity_dt)
    if current_f1_score_dt > max_f1_score_dt:
        max_f1_score_dt = current_f1_score_dt
        best_threshold_f1_dt = threshold
    if current_gmean_score_dt > max_gmean_score_dt:
        max_gmean_score_dt = current_gmean_score_dt
        best_threshold_gmean_dt = threshold
    y_pred_decision_tree_optimized = (y_proba_decision_tree >= best_threshold_f1_dt).astype(int)
    accuracy_decision_tree_optimized = accuracy_score(y_test, y_pred_decision_tree_optimized)
    precision_decision_tree_optimized = precision_score(y_test, y_pred_decision_tree_optimized)
    recall_decision_tree_optimized = recall_score(y_test, y_pred_decision_tree_optimized)
    f1_decision_tree_optimized = f1_score(y_test, y_pred_decision_tree_optimized)
    auc_decision_tree_optimized = roc_auc_score(y_test, y_proba_decision_tree)
    fig, axes = plt.subplots(figsize=(14, 6))
    fig.suptitle('Basic Decision Tree Model Performance with Optimized F1-score Threshold', fontsize=16)
    cm_decision_tree_optimized = confusion_matrix(y_test, y_pred_decision_tree_optimized)
    sns.heatmap(cm_decision_tree_optimized, annot=True, fmt='d', cmap='Blues')
    ax.set_title('Optimized DT: Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    ax.set_yticklabels(['No Diabetes', 'Diabetes'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    st.pyplot(fig)
    #metrics
    st.markdown ('#### Optimized Decision Tree Metrics')
    optimized_dt_df = pd.DataFrame({
        'Metric': ['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
        'Optimized Decision Tree': [
        best_threshold_f1_dt,
        accuracy_decision_tree_optimized,
        precision_decision_tree_optimized,
        recall_decision_tree_optimized,
        f1_decision_tree_optimized,
        auc_decision_tree_optimized]})
    st.dataframe(optimized_dt_df.style.format({'Optimized Decision Tree': "{:.4f}"}))

with st.expander('Random Forest'):
    from sklearn.ensemble import RandomForestClassifier
    # unweighted
    random_forest_model = RandomForestClassifier(random_state=42)
    random_forest_model.fit(X_train_processed, y_train)
    y_pred_rf = random_forest_model.predict(X_test_processed)
    y_proba_rf = random_forest_model.predict_proba(X_test_processed)[:, 1]
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    #weighted
    random_forest_weighted_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    random_forest_weighted_model.fit(X_train_processed, y_train)
    y_pred_rf_weighted = random_forest_weighted_model.predict(X_test_processed)
    y_proba_rf_weighted = random_forest_weighted_model.predict_proba(X_test_processed)[:, 1]
    accuracy_rf_weighted = accuracy_score(y_test, y_pred_rf_weighted)
    precision_rf_weighted = precision_score(y_test, y_pred_rf_weighted)
    recall_rf_weighted = recall_score(y_test, y_pred_rf_weighted)
    f1_rf_weighted = f1_score(y_test, y_pred_rf_weighted)
    auc_rf_weighted = roc_auc_score(y_test, y_proba_rf_weighted)
    #confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Basic Random Forest Model Performance', fontsize=16)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Basic RF: Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['No Diabetes', 'Diabetes'])
    axes[0].set_yticklabels(['No Diabetes', 'Diabetes'])
    cm_rf_weighted = confusion_matrix(y_test, y_pred_rf_weighted)
    sns.heatmap(cm_rf_weighted, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Weighted RF: Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_xticklabels(['No Diabetes', 'Diabetes'])
    axes[1].set_yticklabels(['No Diabetes', 'Diabetes'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)
    #metrics
    st.markdown('#### Random Forest Comparison Table')
    rf_metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Basic Random Forest': [accuracy_rf, precision_rf, recall_rf, f1_rf, auc_rf],
    'Weighted Random Forest': [accuracy_rf_weighted, precision_rf_weighted, recall_rf_weighted, f1_rf_weighted, auc_rf_weighted]})
    st.dataframe(
    rf_metrics_df.style.format({
        'Basic Random Forest': "{:.4f}",
        'Weighted Random Forest': "{:.4f}"}))

with st.expander('Optimized Random Forest'):
    random_forest_weighted_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    random_forest_weighted_model.fit(X_train_processed, y_train)
    y_proba_rf_weighted = random_forest_weighted_model.predict_proba(X_test_processed)[:, 1]
    thresholds = np.arange(0, 1.01, 0.01)
    best_threshold_f1_rf = 0
    max_f1_score_rf = 0
    for threshold in thresholds:
        y_pred_thresholded_rf = (y_proba_rf_weighted >= threshold).astype(int)
        current_f1_score = f1_score(y_test, y_pred_thresholded_rf)
        if current_f1_score > max_f1_score_rf:
            max_f1_score_rf = current_f1_score
            best_threshold_f1_rf = threshold
    y_pred_rf_optimized = (y_proba_rf_weighted >= best_threshold_f1_rf).astype(int)
    accuracy_rf_optimized = accuracy_score(y_test, y_pred_rf_optimized)
    precision_rf_optimized = precision_score(y_test, y_pred_rf_optimized)
    recall_rf_optimized = recall_score(y_test, y_pred_rf_optimized)
    f1_rf_optimized = f1_score(y_test, y_pred_rf_optimized)
    auc_rf_optimized = roc_auc_score(y_test, y_proba_rf_weighted)
    fig, axes = plt.subplots(figsize=(14, 6))
    fig.suptitle('Weighted Random Forest Model Performance with Optimized F1-score Threshold', fontsize=16)
    cm_rf_optimized = confusion_matrix(y_test, y_pred_rf_optimized)
    sns.heatmap(cm_rf_optimized, annot=True, fmt='d', cmap='Blues')
    ax.set_title('Optimized RF: Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    ax.set_yticklabels(['No Diabetes', 'Diabetes'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)
    metrics_rf_optimized = pd.DataFrame({
    'Metric': ['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Optimized Random Forest': [
        best_threshold_f1_rf,
        accuracy_rf_optimized,
        precision_rf_optimized,
        recall_rf_optimized,
        f1_rf_optimized,
        auc_rf_optimized]})
    st.markdown('#### Optimized Random Forest Metrics')
    st.dataframe(metrics_rf_optimized.style.format({'Optimized Random Forest': "{:.4f}"}))

with st.expander('LightGBM with Focal Loss'):
    import lightgbm as lgb
    def focal_loss_objective(y_true, y_pred):
        alpha = 0.75  # Weight for the positive class
        gamma = 1.5   # Focusing parameter
        sigmoid_y_pred = 1 / (1 + np.exp(-y_pred))
        epsilon = 1e-9
        p = np.clip(sigmoid_y_pred, epsilon, 1 - epsilon)
        pt = y_true * p + (1 - y_true) * (1 - p)
        pt = np.maximum(epsilon, pt) # Ensure pt is never zero for np.log
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        grad = alpha_t * (p - y_true) * (1 + gamma * np.log(pt)) * (pt**gamma)
        hess = alpha_t * (pt**gamma) * (
            p * (1 - p) * (1 + gamma * np.log(pt)) -
            gamma * (p - y_true)**2 / pt * p * (1 - p))
        return -grad, -hess
    lgbm_focal_model = lgb.LGBMClassifier(objective=focal_loss_objective, random_state=42,
    n_estimators=100, n_jobs=-1)
    lgbm_focal_model.fit(X_train_processed, y_train)
    raw_scores_lgbm_focal = lgbm_focal_model.predict(X_test_processed)
    y_proba_lgbm_focal = 1 / (1 + np.exp(-raw_scores_lgbm_focal))
    y_pred_lgbm_focal = (y_proba_lgbm_focal >= 0.5).astype(int)
    accuracy_lgbm_focal = accuracy_score(y_test, y_pred_lgbm_focal)
    precision_lgbm_focal = precision_score(y_test, y_pred_lgbm_focal)
    recall_lgbm_focal = recall_score(y_test, y_pred_lgbm_focal)
    f1_lgbm_focal = f1_score(y_test, y_pred_lgbm_focal)
    auc_lgbm_focal = roc_auc_score(y_test, y_proba_lgbm_focal)
    fig, axes = plt.subplots(figsize=(14, 6))
    fig.suptitle('LightGBM Model Performance with Custom Focal Loss', fontsize=16)
    cm_lgbm_focal = confusion_matrix(y_test, y_pred_lgbm_focal)
    sns.heatmap(cm_lgbm_focal, annot=True, fmt='d', cmap='Blues')
    ax.set_title('LGBM Focal: Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    ax.set_yticklabels(['No Diabetes', 'Diabetes'])
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('#### Metrics for LightGBM with Focal Loss')
    lgbm_focal_metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'LightGBM (Focal Loss)': [
        accuracy_lgbm_focal,
        precision_lgbm_focal,
        recall_lgbm_focal,
        f1_lgbm_focal,
        auc_lgbm_focal]})
    st.dataframe(lgbm_focal_metrics_df.style.format({'LightGBM (Focal Loss)': '{:.4f}'}))

with st.expander ('Stacking Classifier'):
    base_estimators_stacking = [('log_reg', log_reg_model),               
    ('log_reg_weighted', log_reg_weighted_model),
    ('random_forest_weighted', random_forest_weighted_model)]
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    final_estimator_stacking = LogisticRegression(random_state=42, solver='liblinear')  
    stacking_clf = StackingClassifier(estimators=base_estimators_stacking,
    final_estimator=final_estimator_stacking,
    cv=5, # Number of cross-validation folds to be used for fitting the estimators
    stack_method='predict_proba', n_jobs=-1, passthrough=True)
    stacking_clf.fit(X_train_processed, y_train)
    y_pred_stacking = stacking_clf.predict(X_test_processed)
    y_proba_stacking = stacking_clf.predict_proba(X_test_processed)[:, 1]
    accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
    precision_stacking = precision_score(y_test, y_pred_stacking)
    recall_stacking = recall_score(y_test, y_pred_stacking)
    f1_stacking = f1_score(y_test, y_pred_stacking)
    auc_stacking = roc_auc_score(y_test, y_proba_stacking)
    fig, axes = plt.subplots(figsize=(14, 6))
    fig.suptitle('StackingClassifier Model Performance', fontsize=16)
    cm_stacking = confusion_matrix(y_test, y_pred_stacking)
    sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Blues')
    ax.set_title('StackingClassifier: Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    ax.set_yticklabels(['No Diabetes', 'Diabetes'])
    st.pyplot(fig)
    st.markdown('#### Metrics for Stacking Classifier')
    stacking_metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Stacking Classifier': [
        accuracy_stacking,
        precision_stacking,
        recall_stacking,
        f1_stacking,
        auc_stacking]})
    st.dataframe(
    stacking_metrics_df.style.format({'Stacking Classifier': '{:.4f}'}))

with st.expander('Other Methods Tried'):
    st.markdown('### Balanced Bagging')
    st.info('Balanced bagging is an ensemble method that builds many models on balanced samples so it can better predict rare classes. It had high recall (0.615) but low precision (0.213), with an F1-score of 0.316 and AUC of 0.846. Optimizing the threshold to 0.70 improved the F1-score to 0.427 and balanced precision (0.395) and recall (0.464). Compared to other models, it outperformed basic logistic regression but was still below the optimized logistic regression.')
    st.markdown('### One-Sided Selection (OSS)')
    st.info ('One-Sided Selection (OSS) is an undersampling technique used to handle imbalanced data sets. It removes noisy examples from the majority class. The goal is to reduce class imbalance. OSS showed slight improvements compared to the basic logistic regression (F1: 0.437, Recall: 0.302). Optimizing the classification threshold to 0.23 significantly improved the F1-score to 0.531, balancing precision (0.621) and recall (0.464), while maintaining an AUC of 0.871. It slightly outperformed the optimized logistic regression F1-score, but is still not considered the best as we consider recall more important.')
    st.markdown('### Voting Classifier')
    st.info ('A voting classifier is an ensemble model that combines the prediction of multiple models. After optimizing the threshold to 0.46, the VotingClassifier achieved an F1-score of 0.464, Precision of 0.629, Recall of 0.368, and AUC of 0.875. It is behind on other models we have tried.')

st.markdown('## Feature Engineering')
with st.expander ('New Features'):
    st.write ('Pulse Pressure: the difference between systolic and diastolic blood pressure (SYSBP - DIABP)')
    st.divider()
    st.write ('Mean Arterial Pressure: calculated as DIABP + 1/3 * (SYSBP - DIABP) to estimate average blood pressure')
    st.divider()
    st.write ('BMI Categories: converted the continuous BMI into groups: Underweight (<18.5), Normal (18.5–25), Overweight (25–30), and Obese (>=30)')
with st.expander ('Results'):
    st.info('After adding these new features, we retrained our models but did not observe significant improvements in performance metrics. Therefore, we proceeded with our original feature set.')

st.markdown('## Conclusion')
with st.expander ('Best Model'):
    st.info('The optimized logistic regression model is the best-performing model in this study. With an F1-score of 0.529 it score slightly lower than the optimized one-sided selection model (F1=0.531), but its recall was higher (0.47). It is the most reliable and well-balanced model for this imbalanced classification task.')

st.markdown('## Limitations & Future Work')
with st.expander ('Limitations'):
    st.write ('1. Data Imbalance: The dataset has a significant class imbalance, which can bias model performance towards the majority class.')
    st.divider()
    st.write ('2. Feature Limitations: The available features may not capture all relevant factors influencing diabetes risk, such as genetic predisposition or lifestyle factors that are not recorded in the dataset.')
    st.divider()
    st.write ('3. Model Generalizability: The models were trained and tested on a specific dataset, which may limit their applicability to other populations or settings.')
with st.expander ('Future Work'):
    st.info('We could further focus on deeper tuning of the excisting models to improve their performance. Also, validation the models on external data sets would help improve their stability and generalization. Finally, exploring additional features or data sources could enhance the models ability to predict diabetes risk.')
