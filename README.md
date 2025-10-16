## Overview : 
    
---

This project conducts a comprehensive analysis of YouTube channels within the popular "cat" niche to identify the key factors that drive video virality. By examining data from my personal channel (faroun) and two competitors (awalf, PetSmile), this analysis moves from broad performance metrics to specific, actionable content strategies.

## 📊 Key Insights & Findings

---

### 1. The analysis reveals that video success in this niche is not random but is strongly correlated with specific, replicable content patterns.

    Optimal Video Duration is 50-60 Seconds:

    Top-performing channels (awalf and faroun) achieve most of their viral hits (videos with >1 million views) with content that is between 50 and 60 seconds long.

    My channel, faroun, shows a particularly strong correlation, with viral success almost exclusively clustered in the 57-60 second range. Unviral content on this channel has a much wider and inconsistent duration.

    The competitor PetSmile focuses on shorter content, a strategy that appears less effective for achieving high-level virality.

### 2. Content Style is a Primary Driver of Virality:

    - Emotional and Story-driven narratives vastly outperform Fact-based content. Despite "Fact-based" being the most common content style in the niche, it is the least likely to go viral for the top channels.

    - For faroun, Storytelling accounts for nearly all viral success, highlighting a clear formula.

    - For awalf, Emotional content is the primary driver of virality, followed by Story-driven videos.
    ### 3. Performance Varies Significantly by Channel:
    Performance Varies Significantly by Channel:

    - awalf demonstrates the strongest overall performance, leading in both total and average views per video. This suggests a highly effective content and engagement strategy.

## ⚙️ Project Workflow

    ---

The analysis follows an iterative process where each step informs the next:

    1. **Data Collection & Loading**: Data, including video metrics and transcripts, was collected for three channels in the cat niche.

    2. **Exploratory Data Analysis (EDA)**: Initial analysis was performed to understand data distributions, compare channel performance (total views, average views), and identify top-performing videos.

    3. **Data Cleaning**: The raw data was preprocessed by converting data types (e.g., upload_date to datetime), removing irrelevant columns (video_id), and ensuring data integrity.

    4. **Content Categorization & Modeling**:

    - A sample of video transcripts was programmatically labeled as "Story", "Emotional", or "Fact-based" using the Google Gemini API.

    - A LogisticRegression model was trained on this labeled data to categorize the entire dataset. The model's performance confirmed a severe class imbalance, reinforcing that "Emotional" and "Story" content are rare but impactful.

    - Techniques like SMOTE and other models (RandomForestClassifier) were explored to address the imbalance, further highlighting - that the linguistic signals for "Story" and "Emotional" content are nuanced beyond simple word frequency.

     5. **Insight Generation**: The cleaned and categorized data was visualized to derive the key findings detailed above.

## 🚀 Setup and Installation

    ---

To run this project locally, follow these steps:
    1. **Clone the repository**:
        ```bash
        git clone https://github.com/anas-elhilali/Youtube-faroun-analytics
        cd youtube-virality-analysis
        ```
    2. **Create a virtual environment and install dependencies**:
        ```bash
            python -m venv venv
            source venv\Scripts\activate   # On Windows use `venv\Scripts\activate`, Mac/Linux `venv/bin/activate`
            pip install -r requirements.txt
        ```
    3. **Set up environment variables**:
    - This project uses the Google Gemini API via Vertex AI for content labeling. You will need a Google Cloud service account with the necessary permissions.
    - Create a .env file in the root directory and add your credentials:
    
        ```
        GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
        PROJECT_ID="your-gcp-project-id"
        LOCATION="your-gcp-project-location"    
        ```
        