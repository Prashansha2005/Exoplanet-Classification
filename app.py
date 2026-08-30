import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64
import joblib
import hashlib
import io
from pathlib import Path
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Exoplanet Detection🪐",
    page_icon="🪐",
    layout="wide"
)
with open("exoplanet_image.png", "rb") as img_file:
    encoded = base64.b64encode(img_file.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(
                rgba(0, 0, 0, 0.3),
                rgba(0, 0, 0, 0.3)
            ),
            url("data:image/jpeg;base64,{encoded}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #071a2f;
}
</style>
""", unsafe_allow_html=True)





# ============================================================
# HEADER
# ============================================================

st.title("🪐 Exoplanet Detection")
st.caption("NASA Space Apps Challenge Project")

st.info(
    """
    Upload an exoplanet dataset, explore its features,
    and prepare the data for machine learning.
    """
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "exo" not in st.session_state:
    st.session_state.exo = None

if "binary_converted" not in st.session_state:
    st.session_state.binary_converted = False


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛰 Contents")

st.sidebar.markdown(
    """
    ### 📂 Dataset
    - Upload Dataset
    - Data Overview
    - Data Exploration

    ### 📊 Data Science
    - Feature Distribution
    -Exploratory Visualizations
    - Outlier Detection
    - Feature Relationships
    - Correlation Analysis

    ### 🤖 Machine Learning
    - Model Comparison
    - ROC Curves
    - Model Performance
 """
 
    
)


st.sidebar.divider()

st.sidebar.caption(
    "🪐 Exoplanet Detection\n"
    "NASA Space Apps Challenge Project"
)

st.subheader("Don't have any CSV files to test? Try these!")
col1, col2 = st.columns(2)
valid_path = Path("demo_csv") / "cumulative_2026.06.21_12.57.11.csv"
valid_data = valid_path.read_bytes()
with col1:
    st.write("Contains clean data matching the required schema.")
    st.download_button(
    label="Download valid Sample",
    data=valid_data,
    file_name="demo_csv_right.csv",
    mime="text/csv",
    type="secondary"
)

invalid_path = Path("demo_csv") / "cumulative_2026.06.23_01.39.42.csv"
# Read the actual CSV bytes so the download contains CSV data.
invalid_data = invalid_path.read_bytes()
with col2:
 st.write("Contains missing column features for testing validation")
 st.download_button(
    label="Download Invalid Sample",
    data=invalid_data,
    file_name="demo_csv_wrong.csv",
    mime="text/csv",
    type="secondary"
)
st.caption("Download either CSV to test the validation flow.")
st.divider()
# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)


# ============================================================
# MAIN APP
# ============================================================


# Load these once before the upload logic.
if "df" not in st.session_state:
    st.session_state.df = None

if "list_of_missing_columns" not in st.session_state:
    st.session_state.list_of_missing_columns = []


if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

if "df" not in st.session_state:
    st.session_state.df = None

if "list_of_missing_columns" not in st.session_state:
    st.session_state.list_of_missing_columns = []

if uploaded_file:

    file_bytes = uploaded_file.getvalue()
    current_signature = hashlib.md5(file_bytes).hexdigest()

    if current_signature != st.session_state.file_signature:

       
        st.session_state.file_signature = current_signature
        st.session_state.df = pd.read_csv(
            io.BytesIO(file_bytes),
            comment="#",
            on_bad_lines="skip"
        )
        st.session_state.exo = st.session_state.df.copy()
        st.session_state.list_of_missing_columns = []
        st.session_state.output = None

        st.success("✅ New file uploaded successfully!")

    df = st.session_state.df

    feature_columns = joblib.load("feature_columns.pkl")

    missing_columns = [
        col for col in feature_columns
        if col not in df.columns
    ]

    list_of_missing_columns=[]

    if missing_columns:
        missing_text = "\n".join(missing_columns)

        st.warning(
            "⚠️ The uploaded dataset is missing required feature columns."
        )

        st.error(
            f"Missing expected feature columns:\n{missing_text}"
        )
        st.session_state.missing_columns = missing_columns
    else:
        st.success(
            "✅ The uploaded dataset matches the expected feature columns."
        )
        st.session_state.missing_columns = []

    st.divider()
    
    st.divider()


    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.header("📌 Dataset Overview")
    preview_df = df.head().astype(str)
    st.dataframe(
        df.head(),
        width="stretch"
    )


    st.info(
        f"""
        This dataset contains **{df.shape[0]} rows**
        and **{df.shape[1]} columns**.
        """
    )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )


    st.divider()


    # ========================================================
    # DATA EXPLORATION
    # ========================================================

    st.header("🔍 Data Exploration")

    tab_dataset, tab_columns, tab_statistics, tab_missing, tab_types = st.tabs(
        [
            "C Dataset",
            "📑 Columns",
            "📊 Statistics",
            "🔍 Missing Values",
            "📂 Data Types"
        ]
    )


    # --------------------------------------------------------
    # DATASET TAB
    # --------------------------------------------------------

    with tab_dataset:

        st.dataframe(
            df,
            use_container_width=True
        )


    # --------------------------------------------------------
    # COLUMNS TAB
    # --------------------------------------------------------

    with tab_columns:

        st.write(
            "### Dataset Columns"
        )

        st.write(
            df.columns.tolist()
        )


    # --------------------------------------------------------
    # STATISTICS TAB
    # --------------------------------------------------------

    with tab_statistics:

        st.dataframe(
            df.describe(),
            use_container_width=True
        )




    # ========================================================
    # MISSING VALUES
    # ========================================================
    with tab_missing:
        st.header("🔍 Missing Values")

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:

         st.success(
            "✅ No missing values found!"
         )

        else:

         st.dataframe(
            missing.sort_values(
                ascending=False
            ),
            use_container_width=True
        )


    # ========================================================
    # DATA TYPES
    # ========================================================

    with tab_types:
        st.header("📂 Data Types")

        types = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str)
        }
        )

        st.dataframe(
        types,
        use_container_width=True
        )



    st.divider()


    # ========================================================
    # FEATURE RELATIONSHIP
    # ========================================================

    st.header("📊 Relationship Between Two Features")

    numeric_df = df.select_dtypes(
        include="number"
    ).columns


    if len(numeric_df) >= 2:

        col1, col2 = st.columns(2)

        with col1:

            x_axis = st.selectbox(
                "Select X-axis",
                numeric_df,
                key="x_axis"
            )

        with col2:

            y_axis = st.selectbox(
                "Select Y-axis",
                numeric_df,
                index=1,
                key="y_axis"
            )


        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            title=f"{y_axis} vs {x_axis}",
            opacity=0.7
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Not enough numeric columns for a scatter plot."
        )


    # ========================================================
    # EDA VISUALIZATIONS
    # ========================================================

    st.header("📊 Exploratory Visualizations")



    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    st.subheader("📈 Feature Correlation Heatmap")
    st.caption("Explore linear relationships between numerical features.")
    numeric_df = df.select_dtypes(
            include="number"
        )

    if numeric_df.shape[1] > 1:

            selected_features = st.multiselect(
                "Select Features",
                numeric_df.columns,
                default=list(
                    numeric_df.columns[:10]
                )
            )

            if len(selected_features) >= 2:

                corr = numeric_df[
                    selected_features
                ].corr()

                fig, ax = plt.subplots(
                    figsize=(10, 8)
                )

                sns.heatmap(
                    corr,
                    cmap="coolwarm",
                    annot=True,
                    square=True,
                    linewidths=0.5,
                    cbar=True,
                    ax=ax
                )

                ax.set_title(
                    "Feature Correlation Heatmap"
                )

                st.pyplot(
                    fig
                )

            else:

                st.info(
                    "Please select at least two features."
                )

    else:

            st.warning(
                "Not enough numeric columns."
            )


    # --------------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------------

    st.subheader("📈 Histogram")
    st.caption("Use the histogram to inspect the distribution and skewness of a feature.")
    numeric_df = df.select_dtypes(
            include="number"
        )

    if numeric_df.shape[1] > 0:

            selected_feature = st.selectbox(
                "Select Feature",
                numeric_df.columns,
                key="histogram_feature"
            )

            fig = px.histogram(
                df,
                x=selected_feature,
                nbins=30,
                title=f"Histogram of {selected_feature}",
                opacity=0.7
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

            st.warning(
                "No numeric columns available for histogram."
            )


    # --------------------------------------------------------
    # BOXPLOT
    # --------------------------------------------------------

    st.subheader("🔍 Boxplot")

    numeric_df = df.select_dtypes(
            include="number"
        )

    if numeric_df.shape[1] > 0:

            selected_feature = st.selectbox(
                "Select Feature",
                numeric_df.columns,
                key="boxplot_feature"
            )

            fig = px.box(
                df,
                y=selected_feature,
                title=f"Boxplot of {selected_feature}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

            st.warning(
                "No numeric columns available for boxplot."
            )


    st.subheader("📦 Outlier Detection")

    method = st.selectbox(
        "Detection Method",
        ["IQR", "Z-Score"]
          )

    data = df[selected_feature].dropna()

    if method == "IQR":

         

          Q1 = data.quantile(0.25)
          Q3 = data.quantile(0.75)

          IQR = Q3 - Q1

          lower_bound = Q1 - 1.5 * IQR
          upper_bound = Q3 + 1.5 * IQR

          outlier_mask = (
          (data < lower_bound) |
          (data > upper_bound)
          )

          outliers = data[outlier_mask]

          st.write(f"**Q1:** {Q1:.3f}")
          st.write(f"**Q3:** {Q3:.3f}")
          st.write(f"**IQR:** {IQR:.3f}")

          st.write(f"**Lower bound:** {lower_bound:.3f}")
          st.write(f"**Upper bound:** {upper_bound:.3f}")

          st.write(
           f"**Potential outliers:** {len(outliers):,} "
           f"({len(outliers) / len(data) * 100:.2f}%)"
           )

    else:

         mean = data.mean()
         std = data.std()
  
         z_scores = (data - mean) / std

         outlier_mask = z_scores.abs() > 3

         outliers = data[outlier_mask]

         st.write(
         f"**Potential outliers (|z| > 3):** {len(outliers):,}"
         )
    # ============================================================
# MODEL COMPARISON
# ============================================================

    st.divider()

    st.header("🤖 Model Comparison")

    st.write(
    """
    Comparison of the machine learning models evaluated for
    exoplanet classification.
    """
    )

# ------------------------------------------------------------
# MODEL METRICS
# ------------------------------------------------------------

    model_metrics = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "HistGradientBoosting"
    ],

    "Accuracy": [
        0.93,
        0.97,
        0.98
    ],

    "Precision": [
        0.93,
        0.97,
        0.97
    ],

    "Recall": [
        0.93,
        0.97,
        0.98
    ],

    "F1-Score": [
        0.93,
        0.97,
        0.98
    ],

    "ROC-AUC": [
        0.978,
        0.994,
        0.997
    ]
    })

# ------------------------------------------------------------
# METRICS TABLE
# ------------------------------------------------------------

    st.subheader("📋 Performance Metrics")

    st.dataframe(
    model_metrics.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-Score": "{:.2%}",
        "ROC-AUC": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
    )


# ------------------------------------------------------------
# PERFORMANCE CHART
# ------------------------------------------------------------

    st.subheader("📊 Model Performance")

    metrics_to_plot = st.multiselect(
    "Select metrics to compare",
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],
    default=[
        "Accuracy",
        "F1-Score"
    ]
    )

    if metrics_to_plot:

     comparison_df = model_metrics.melt(
        id_vars="Model",
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score"
     )

     fig = px.bar(
        comparison_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        range_y=[0, 1],
        title="Model Performance Comparison"
     )

     fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="inside"
     )

     fig.update_layout(
        template="plotly_dark",
        yaxis_title="Score",
        xaxis_title="Model"
     )

     st.plotly_chart(
        fig,on_select="ignore",height="stretch",
        use_container_width=None
     )

    st.divider()
    data = np.load("auc_data.npz")
    fpr_lr = data["fpr_lr"]
    tpr_lr = data["tpr_lr"]
    fpr_rf = data["fpr_rf"]
    tpr_rf = data["tpr_rf"]
    fpr_hgb = data["fpr_hgb"]
    tpr_hgb = data["tpr_hgb"]
    

    st.subheader("📈 ROC Curve Comparison")

    fig = px.line(
    x=fpr_lr,
    y=tpr_lr,
    labels={
        "x": "False Positive Rate",
        "y": "True Positive Rate"
    },
    title="ROC Curves"
)

    fig.add_scatter(
    x=fpr_rf,
    y=tpr_rf,
    mode="lines",
    name="Random Forest (AUC = 0.994)"
)

    fig.add_scatter(
    x=fpr_hgb,
    y=tpr_hgb,
    mode="lines",
    name="HistGradientBoosting (AUC = 0.997)"
)

    fig.add_scatter(
    x=[0, 1],
    y=[0, 1],
    mode="lines",
    name="Random Classifier"
)

    fig.update_layout(
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1])
)

    st.plotly_chart(
    fig,
    use_container_width=True
    )
    st.info("The ROC curve illustrates the performance of the classification models. The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the model. The area under the curve (AUC) provides a single measure of overall model performance.")
# ------------------------------------------------------------
# BEST MODEL
# ------------------------------------------------------------

    best_model = model_metrics.loc[
    model_metrics["Accuracy"].idxmax()
    ]

    st.success(
    f"🏆 **Best performing model by accuracy: "
    f"{best_model['Model']} ({best_model['Accuracy']:.0%})**"
    )


    st.divider()

    st.subheader("🔍Time to predict!")
    st.caption("prediction for the csv file uploaded...")
    
    api_url = os.getenv("API_URL", "http://localhost:8080")
    button=st.button("Send to API for prediction")
    list_of_missing_columns=st.session_state.missing_columns
    if "output" not in st.session_state:
      st.session_state.output = None
      
    if button: 
                        
                        files={"file":(uploaded_file.name,uploaded_file.getvalue(),uploaded_file.type)}
                        try:
                          with st.spinner("Sending to API.."):
                                response=requests.post(api_url,files=files)
                                if response.status_code==200:
                                      st.info("API received the CSV successfully")
                                      output= response.json()
                                      st.session_state.output=output
                                      
                                       
                                       
                                else:
                                         if list_of_missing_columns:
                                              st.error("Columns are missing to perform predictions..")
                                              st.stop()
                                         st.error(f"Failed to send the csv to api..{response.status_code}")
                                         st.write(response.text) 
                        except Exception as e:
                                                     st.error(f"An error occurred: {e}")
   
                                           
    if st.session_state.get("output") is not None:
       output = st.session_state.output
       raw_predictions = output["predictions"]

    
       total_exo = sum(prediction == 1 for prediction in raw_predictions)
       total_non_exo = len(raw_predictions) - total_exo

       

   
       col1, col2, col3 = st.columns(3)

       col1.metric("Rows Processed", output["rows_processed"])
       col2.metric("Exoplanets Detected", total_exo)
       col3.metric("Not Exoplanets", total_non_exo)

       st.subheader("Prediction Summary")

   
       summary_df = pd.DataFrame({
        "Category": ["Exoplanet", "Not Exoplanet"],
        "Count": [total_exo, total_non_exo]
      }).set_index("Category")

       st.bar_chart(summary_df)

    
       labeled_predictions = [
        "Exoplanet" if prediction == 1 else "Not Exoplanet"
        for prediction in raw_predictions
       ]

       results_df = pd.DataFrame({
        "Prediction": labeled_predictions
      })

       st.subheader("Detailed Predictions")

    
       st.dataframe(
        results_df,
        height=400,
        use_container_width=True,
        hide_index=True
       )

    
       download_data = results_df.to_csv(index=False)

       st.download_button(
        label="Download predictions as CSV",
        data=download_data,
        file_name="predictions.csv",
        mime="text/csv"
       )
       st.success("Prediction completed successfully!")                              
                   

else:

      st.info(
        "👆 Upload a CSV file to begin exploring the dataset and make predictions!"
     )