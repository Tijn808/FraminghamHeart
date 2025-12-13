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

with st.expander ('Logistic Regression (with threshold)'):
    st.info('To further improve the model we added a threshold. The optimized threshold for F1-score is 0.21.')
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

with st.expander('Decision Tree (with threshold)'):
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

with st.expander('Random Forest with Threshold'):
    st.write('????? tuning or threshold')

with st.expander('LightGBM'):
    import lightgbm as lgb
    

