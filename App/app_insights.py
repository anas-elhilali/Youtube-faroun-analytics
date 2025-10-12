import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import plotly.express as px
st.title("📊 Viral Content Analysis Dashboard")
st.subheader("Overview")
st.write("""
This dashboard explores what makes content go viral using real-world data from my channel (`faroun`) and competitors.
The analysis follows an iterative process:
1. **Data Collection / Loading:** Load channel data, transcripts, and engagement metrics.
2. **Exploratory Data Analysis (EDA):** Understand distributions, trends, and relationships.
3. **Data Cleaning / Preprocessing:** Fix missing values, standardize formats, prepare features.
4. **Modeling:** Build predictive models for labeling.
5. **Iteration:** Re-explore the data and refine models based on insights, charts, and patterns.
""")
faroun_data = pd.read_csv("Youtube-faroun-analytics/data/niche_data/Faroun_cats_transcription.csv")
awalf_data = pd.read_csv("Youtube-faroun-analytics/data/niche_data/awalefofficial_transcription.csv")
petsmile_data = pd.read_csv("Youtube-faroun-analytics/data/niche_data/Smiling-Pet_transcription.csv")


st.markdown("""
## 📊 Exploratory Data Analysis (EDA) """)
def eda_data_niche(data):
    st.write("Data set Info")
    st.write(data.head())
    print("------------------------------------")
    st.write("\n Descbtion of the data")
    st.write(data.describe())
    print("------------------------------------")
    st.write("data info")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.code(s, language="bash")
st.markdown("#### PetSmile") 
eda_data_niche(petsmile_data)
st.markdown("#### awalf") 
eda_data_niche(awalf_data)
st.markdown("#### faroun") 
eda_data_niche(faroun_data)
def clean_data(data):
    data = data.drop('video_id', axis=1)
    data['upload_date'] = pd.to_datetime(data['upload_date'], format='%Y%m%d')
    data = data.drop_duplicates()
    return data





def top_5_videos(data, channel_name):
    data_top = data.nlargest(5, 'view_count')
    video_titles_rtl = [get_display(arabic_reshaper.reshape(title)) for title in data_top['title']]
    views_million = data_top['view_count'] / 1_000_000
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(video_titles_rtl, views_million, color='blue')
    ax.set_xlabel("Views (Millions)")
    ax.set_ylabel("Video Title")
    ax.set_title(f"Top 5 Videos for {channel_name}")
    st.pyplot(fig)

top_5_videos(faroun_data, "Faroun")
top_5_videos(awalf_data, "Awalf")
top_5_videos(petsmile_data, "Petsmile")


creators_channels_total_views_list = [
    {'name': 'faroun', 'total_views': (faroun_data['view_count'] / 1_000_000).sum()},
    {'name': 'awalf', 'total_views': (awalf_data['view_count'] / 1_000_000).sum()},
    {'name': 'petsmile', 'total_views': (petsmile_data['view_count'] / 1_000_000).sum()},
]

creators_channels_total_views = pd.DataFrame(creators_channels_total_views_list)
fig = px.bar(
    creators_channels_total_views,
    x='name',
    y='total_views',
    color='name',
    text='total_views',
    labels={'name': 'Channel', 'total_views': 'Total Views (M)'},
    title='Total Views per Channel'
)

fig.update_traces(texttemplate='%{text:.2f} M', textposition='outside')
fig.update_layout(
    xaxis_title="Channel Name",
    yaxis_title="Total Views (Millions)",
    title_x=0.5,
    showlegend=False,
    plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

awalf_data['Creator'] = 'awalf'
faroun_data['Creator'] = 'faroun'
petsmile_data['Creator'] = 'petsmile'

all_data = pd.concat([awalf_data  , faroun_data , petsmile_data] , ignore_index=True)

def viralty_duration_1( data ):
    fig, ax = plt.subplots(figsize=(10,5))
    data['category'] = data['view_count'].apply(lambda x: 'Viral' if x > 1000000 else 'Unviral')

    sns.boxplot(x = 'Creator' ,  y ='duration' , hue='category' , data = data )
    plt.title(f' : Video Duration: Viral vs Unviral')
    st.pyplot(fig)

# %%
viralty_duration_1(all_data)

import vertexai
from vertexai.generative_models import GenerativeModel
import os 
from dotenv import load_dotenv
load_dotenv()
all_data_labeled = all_data
if os.path.exists('Youtube-faroun-analytics/notebook/all_data_labeled.csv'): 
    all_data_labeled = pd.read_csv("Youtube-faroun-analytics/notebook/all_data_labeled.csv")

else:
    SERVICE_ACCOUNT_KEY_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = os.getenv("LOCATION")

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    captions = [caption for caption in all_data_labeled['transcript']]

    for idx , row in all_data_labeled.iterrows():
        caption = row['transcript']  
        model = GenerativeModel("gemini-2.5-flash")
        
        prompt=f"""Act as a script categorizer. Categorize the following script as **Story**, **Emotional**, or **Fact-based**.  
        Return **only the category, one word, nothing else**.  
                Script: "{caption}"
                """
        response = model.generate_content([prompt])

        print(caption)
        print(response.text)
        all_data_labeled.loc[ idx , 'transcript_category'] = response.text.strip()
    all_data_labeled.to_csv("Youtube-faroun-analytics/notebook/all_data_labeled.csv" , index=False)


# %%
def creators_category_transcirpt(data):
    fig , ax = plt.subplots(figsize=(10,5))
    
    sns.countplot(data = data , x='Creator' , hue='transcript_category')
    st.pyplot(fig)
creators_category_transcirpt(all_data_labeled)

def creators_category_transcirpt_viral(data):
    fig , ax = plt.subplots(figsize=(10,5))
    viral_data = data[data['is_viral'] == True]
    sns.countplot(data = viral_data , x='Creator' , hue='transcript_category' )
    st.pyplot(fig)
creators_category_transcirpt_viral(all_data_labeled)

def creators_category_transcirpt_viral(data):
    fig , ax = plt.subplots(figsize=(10,5))
    viral_data = data[data['is_viral'] == True]
    sns.histplot(data = viral_data , x='Creator' , hue='transcript_category' , multiple="fill" , stat='percent')
    st.pyplot(fig)
creators_category_transcirpt_viral(all_data_labeled)