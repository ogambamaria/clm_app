import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import classification_report

# Set the configuration for the page
st.set_page_config(page_title='CLM Prediction App', layout='wide')
st.title('CLM Prediction App')

# Sidebar description and links
st.sidebar.header("About the App")
st.sidebar.info("This Streamlit app is designed to load, preprocess, visualize, and model data using CatBoost with tuned parameters for prediction.")
# st.sidebar.markdown("""
# See the data preparation and model development steps: [GitHub Repository](https://github.com/ogambamaria/mod5_poa)
# """)
st.sidebar.markdown("""
More detailed insights are available on this [PowerBI Dashboard](https://app.powerbi.com/view?r=eyJrIjoiZWEwNjIxZWYtYTFjNS00Y2U5LTk3MDQtZDczNzIxMDRjODJiIiwidCI6ImE0NjIyOWM3LTIxZDEtNDE3ZC1hMWNiLTE4NTdhMDdkMjc2NSIsImMiOjh9)
""", unsafe_allow_html=True)

st.write('This app is designed to predict patient satisfaction based on the data provided. The model is trained using CatBoost with tuned parameters.')

# Function to load the model
@st.cache_data()
def load_model():
    return joblib.load('catboost_model_clm.pkl')

model = load_model()

# Load example data
@st.cache_data()
def load_example_data():
    return pd.read_csv('data/saved_data.csv', index_col=0, low_memory=False)

example_data = load_example_data()

# User input for data upload
# uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)
#     st.write('Uploaded Data Preview:', input_df.head())
# else:
input_df = example_data
    # st.write('Using example data. You can also upload your own data.')

# Button to make predictions
if st.button('Predict'):
    if 'ServiceSatisfaction' in input_df.columns:
        predictions = model.predict(input_df.drop(['ServiceSatisfaction'], axis=1))
        input_df['Predictions'] = predictions
        st.write('Predictions:', input_df.drop(['ServiceSatisfaction'], axis=1))

        # Aggregate and visualize prediction counts
        prediction_counts = pd.DataFrame(predictions, columns=['Predictions']).value_counts().reset_index(name='Counts')
        fig_count = px.bar(prediction_counts, x='Predictions', y='Counts', title='Distribution of Satisfaction', labels={'Predictions': 'Classes', 'Counts': 'Frequency'})
        st.plotly_chart(fig_count)
        st.write('The values 0, 1, and 2 represent dissatisfied, neutral, and satisfied patients, respectively.')

        # Visualize feature importances
        feature_importances = pd.Series(model.feature_importances_, index=model.feature_names_)
        fig = px.bar(feature_importances, x=feature_importances.values, y=feature_importances.index, orientation='h')
        st.plotly_chart(fig)
    else:
        st.error('The data must contain a "ServiceSatisfaction" column for prediction and evaluation.')