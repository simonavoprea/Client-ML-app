import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
from impyute.imputation.cs import mice  # Requires installing impyute package

# Streamlit App
st.title("Client Leasing Data Classifier")
st.write("Upload your CSV file, and select the preferred method to handle missing values.")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Dropdown for missing value method selection
st.sidebar.header("Choose a method to handle missing values")
missing_value_method = st.sidebar.selectbox("Handle NULL",
    ("fillna - zero or unknown", "fillna - mean or mode", "bfill", "ffill", "interpolate", "mice")
)

if uploaded_file:
    # Reload data for each selection to ensure it's fresh
    df = pd.read_csv(uploaded_file)


    st.subheader("Missing Values in Percentage")

    # Calculate the percentage of missing values for each column
    missing_percentage = df.isnull().mean() * 100
    missing_percentage = missing_percentage[missing_percentage > 0]  # Only show columns with missing values

    if not missing_percentage.empty:
        # Convert to DataFrame for better display
        missing_df = pd.DataFrame(missing_percentage, columns=["Percentage of Missing Values"])
        st.table(missing_df)  # Display as a static table
    else:
        st.write("No missing values in the dataset.")


    st.subheader("Missing Values Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap = 'summer', yticklabels=False)
    st.pyplot(plt)

    # Drop unnecessary columns
    cols_to_drop = ["ID_CLIENT", "NUME_CLIENT", "CATEGORIE", "DESCRIERE"]
    df.drop(columns=cols_to_drop, errors="ignore", inplace=True)
    
    # Data Transformation
    df['DATA'] = pd.to_datetime(df['DATA'], errors='coerce')
    df['LUNA'] = df['DATA'].dt.month
    df['AN'] = df['DATA'].dt.year
    df.drop("DATA", axis=1, inplace=True)
    
    # Categorize professions not in the specified list
    listOfProf = ['muncitor necalificat', 'profesor', 'agricultor', 'asistent medical', 'barman', 'economist', 'inginer', 'medic', 'pensionar']
    df['PROFESIA'] = df['PROFESIA'].apply(lambda x: x.lower() if x.lower() in listOfProf else 'ALTA PROFESIE')
    
    # Apply the selected missing value handling method
    if missing_value_method == "fillna - zero or unknown":
        # Fill numerical columns with the mean
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(0, inplace=True)
            
        # Fill categorical/string columns with the mode (most frequent value)
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna("Unknown", inplace=True)
    elif missing_value_method == "fillna - mean or mode":
        # Fill numerical columns with the mean
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].mean(), inplace=True)
            
        # Fill categorical/string columns with the mode (most frequent value)
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    elif missing_value_method == "bfill":
        df.fillna(method='bfill', inplace=True)
    elif missing_value_method == "ffill":
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
    elif missing_value_method == "interpolate":
        df.interpolate(method='linear', limit_direction='both', inplace=True)
    elif missing_value_method == "mice":
        # Define np.float temporarily to avoid deprecation errors
        np.float = float
        df_numeric = df.select_dtypes(include=[np.number])
        imputed_df = mice(df_numeric.values)
        df[df_numeric.columns] = pd.DataFrame(imputed_df, columns=df_numeric.columns)
    

    # Display processed data
    st.subheader("Processed Data Sample")
    st.write(df.head())
    
    # Encoding categorical columns
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))
    
    # Define features and target
    X = df.drop('PROBABILITATE_CONTRACTARE_N', axis=1, errors='ignore')
    y = df['PROBABILITATE_CONTRACTARE_N'] if 'PROBABILITATE_CONTRACTARE_N' in df.columns else None
    
    # Feature Selection
    if y is not None:
        selector = SelectKBest(f_regression, k=5)
        X_selected = selector.fit_transform(X, y)
        X = pd.DataFrame(X_selected, columns=X.columns[selector.get_support()])
    
        # Standardization
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        # Model Training and Evaluation
        model = LogisticRegression(max_iter=50, multi_class='ovr', penalty='l2', solver='lbfgs')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Display Evaluation Metrics
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.subheader("Model Evaluation Metrics")
        st.write("Confusion Matrix:")

        # Create the heatmap
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap = 'summer', cbar=False, ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix Heatmap")

        # Display the plot in Streamlit
        st.pyplot(fig)
        
        from sklearn.metrics import classification_report

        # Generate the classification report as a dictionary
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Convert the dictionary to a DataFrame
        report_df = pd.DataFrame(report_dict).transpose()

        # Round the metrics for better readability
        report_df['precision'] = report_df['precision'].apply(lambda x: round(x, 2))
        report_df['recall'] = report_df['recall'].apply(lambda x: round(x, 2))
        report_df['f1-score'] = report_df['f1-score'].apply(lambda x: round(x, 2))
        report_df['support'] = report_df['support'].astype(int)

        # Display the DataFrame in Streamlit
        st.subheader("Classification Report")
        st.dataframe(report_df)
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        
        # ROC Curve
        st.subheader("ROC Curve")
        model_probs = model.predict_proba(X_test)[:, 1]
        ns_probs = [0 for _ in range(len(y_test))]
        ns_auc = roc_auc_score(y_test, ns_probs)
        model_auc = roc_auc_score(y_test, model_probs)
        st.write(f"Model ROC AUC: {model_auc:.3f}")
        
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)
        
        plt.figure()
        plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
        plt.plot(model_fpr, model_tpr, marker=".", label="Logistic Regression")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(plt)
    else:
        st.write("Target variable 'PROBABILITATE_CONTRACTARE_N' not found in the dataset.")
