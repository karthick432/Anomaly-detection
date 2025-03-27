import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Streamlit UI
st.title("Network Anomaly Detection System")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Debug: Raw Dataset Before Processing")
    st.dataframe(df.head(10))

    # Convert only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    st.subheader("Debug: After Conversion to Numeric")
    st.dataframe(df.head(10))

    # Drop NaN values only in numeric columns
    df.dropna(subset=numeric_cols, inplace=True)

    if df.empty:
        st.error("Error: Dataset is empty after removing NaN values!")
    else:
        st.subheader("Debug: After Dropping NaN Values")
        st.dataframe(df.head(10))

        # Drop duplicate & constant columns
        df = df.loc[:, df.nunique() > 1]

        if df.empty:
            st.error("Error: Dataset is empty after removing constant columns!")
        else:
            st.subheader("Debug: After Dropping Duplicate & Constant Columns")
            st.dataframe(df.head(10))

            # Anomaly Detection (Simple Threshold-Based)
            threshold = df['latency'].quantile(0.95)  # 95th percentile
            anomalies_df = df[df['latency'] > threshold]

            anomaly_count = len(anomalies_df)

            st.markdown(f"## Number of Anomalies: **{anomaly_count}**")

            if anomaly_count > 0:
                st.subheader("Sample Anomalies:")
                st.dataframe(anomalies_df.head(10))

                # Plot Anomalies
                st.subheader("Anomaly Visualization")
                fig = px.scatter(anomalies_df, x=anomalies_df.index, y="latency",
                                 color_discrete_sequence=['red'], title="Detected Anomalies",
                                 labels={"index": "Index", "latency": "Latency"})
                st.plotly_chart(fig)
            else:
                st.success("No anomalies detected!")
