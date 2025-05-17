import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Drug Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
        .main { background-color: #f4f4f4; }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stMetric {
            background-color: #e8f0fe !important;
            padding: 10px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Load data
df = pd.read_csv('your_processed_data.csv')  # Ensure this file is in your GitHub repo

st.title("💊 Drug Sentiment Analysis Dashboard")
st.markdown("Analyze patient experiences using sentiment scores from drug reviews to assist in recommending suitable medications.")

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
condition = st.sidebar.selectbox("Select Condition", options=sorted(df['condition'].dropna().unique()))
drug_options = df[df['condition'] == condition]['drugName'].dropna().unique()
selected_drugs = st.sidebar.multiselect("Select Drug(s)", options=sorted(drug_options))

# Filter Data
filtered_df = df[df['condition'] == condition]
if selected_drugs:
    filtered_df = filtered_df[filtered_df['drugName'].isin(selected_drugs)]

# Summary Metrics
st.subheader(f"📈 Summary for: {condition}")
col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(filtered_df))
col2.metric("Average Sentiment", round(filtered_df['sentiment_score'].mean(), 3))
if not filtered_df.empty:
    top_drug = filtered_df['drugName'].value_counts().idxmax()
    col3.metric("Most Reviewed Drug", top_drug)

# Bar Chart: Average Sentiment per Drug
if not filtered_df.empty:
    avg_sentiment = filtered_df.groupby('drugName')['sentiment_score'].mean().sort_values(ascending=False)
    st.subheader("🧪 Average Sentiment Score per Drug")
    fig1 = px.bar(avg_sentiment, title="Average Sentiment by Drug", color=avg_sentiment, color_continuous_scale='Tealgrn')
    st.plotly_chart(fig1, use_container_width=True)

    # Sentiment Label Distribution
    st.subheader("📊 Sentiment Label Distribution")
    sentiment_counts = filtered_df.groupby(['drugName', 'sentiment_label']).size().unstack(fill_value=0)
    fig2 = px.bar(sentiment_counts, barmode='stack', title="Sentiment Label Count per Drug", 
                  labels={"value": "Review Count", "drugName": "Drug"}, 
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig2, use_container_width=True)

    # Word Cloud
    st.subheader("☁️ Word Cloud from Reviews")
    sentiment_to_plot = st.radio("Choose Sentiment for Word Cloud", ['Positive', 'Negative'])
    word_text = " ".join(filtered_df[filtered_df['sentiment_label'] == sentiment_to_plot]['clean_review'].dropna().astype(str))
    
    if word_text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Dark2').generate(word_text)
        fig3, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig3)
    else:
        st.info("No reviews available for the selected sentiment.")

    # Show Data
    if st.checkbox("🗂 Show Raw Data"):
        st.dataframe(filtered_df[['drugName', 'clean_review', 'sentiment_score', 'sentiment_label']])

    # Download Button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Data", data=csv, file_name="filtered_reviews.csv", mime='text/csv')
else:
    st.warning("No data available for the selected filters.")
