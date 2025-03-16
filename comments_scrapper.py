import os
import streamlit as st
import pandas as pd
import time
import json
import concurrent.futures
import matplotlib.pyplot as plt
import plotly.express as px
from collections import defaultdict, Counter
from wordcloud import WordCloud
from apify_client import ApifyClient
from openai import OpenAI  # Using the new OpenAI client style

# Load API keys from environment variables
APIFY_API_KEY = os.getenv("APIFY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------- CONSTANTS & CLIENT INITIALIZATION --------------------
FIND_BUSINESS_ACTOR_ID = "nwua9Gu5YrADL7ZDj"
SCRAPE_REVIEWS_ACTOR_ID = "Xb8osYTtOjlsgI6k9"

client = ApifyClient(APIFY_API_KEY)

# -------------------- HELPER FUNCTIONS --------------------
def center_pyplot(fig):
    """Display a matplotlib figure centered on the page."""
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.pyplot(fig)

def center_plotly(fig):
    """Display a Plotly figure centered on the page."""
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.plotly_chart(fig)

def group_pie_data(values, labels):
    """
    Group pie chart data by identical numeric values.
    If multiple labels share the same numeric value, their labels are combined
    (comma separated) and their values are summed.
    Returns a tuple (new_values, new_labels).
    """
    grouped = defaultdict(int)
    grouped_labels = defaultdict(list)
    for val, lab in zip(values, labels):
        grouped[val] += val
        grouped_labels[val].append(lab)
    # Sort by value (largest first) for consistent display.
    sorted_keys = sorted(grouped.keys(), reverse=True)
    new_values = [grouped[k] for k in sorted_keys]
    new_labels = [", ".join(grouped_labels[k]) for k in sorted_keys]
    return new_values, new_labels

# -------------------- FUNCTION DEFINITIONS --------------------
def find_business(name=None, location=None, business_url=None):
    """Find business details using Apify's search actor or a provided URL."""
    if business_url:
        run_input = {
            "startUrls": [{"url": business_url}],
            "language": "en",
            "maxCrawledPlacesPerSearch": 1,
        }
    else:
        run_input = {
            "searchStringsArray": [name],
            "locationQuery": location,
            "language": "en",
            "maxCrawledPlacesPerSearch": 1,
        }
    start_time = time.time()
    while time.time() - start_time < 90:  # 90 seconds timeout
        try:
            run = client.actor(FIND_BUSINESS_ACTOR_ID).call(run_input=run_input)
            results = list(client.dataset(run["defaultDatasetId"]).iterate_items())
            if results:
                return results[0]
        except Exception as e:
            st.error(f"Error fetching business: {e}")
            return None
        time.sleep(5)
    return None

def scrape_reviews(business_url, max_reviews=5):
    """Scrape reviews using Apify."""
    run_input = {
        "startUrls": [{"url": business_url}],
        "maxReviews": max_reviews,
        "reviewsSort": "newest",
        "language": "en",
    }
    try:
        run = client.actor(SCRAPE_REVIEWS_ACTOR_ID).call(run_input=run_input)
        reviews = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        # Keep time-related columns for temporal analysis
        return pd.DataFrame(reviews) if reviews else None
    except Exception as e:
        st.error(f"Error fetching reviews: {e}")
        return None

def process_comment_seq(row, openai_client):
    """
    Process a single review comment using the new OpenAI client.
    Requests a structured JSON response.
    """
    review_json = row.to_json()
    prompt = f"""
Analyze the following review comment and generate a structured JSON output. Consider the factors below:

- **Comment**: {review_json}

1. **Sentiment Analysis**:  
   Evaluate the emotional tone and assign a score (0: very angry, 1: angry, 2: neutral, 3: happy, 4: very happy).

2. **Intent Analysis**:  
   Determine the review’s intent (0: support, 1: criticism, 2: enquiry, 3: complaint, 4: praise, 5: sarcasm).

3. **Named Entity Recognition (NER)**:  
   Extract from the comment:
   - person_mentioned (list),
   - product_mentioned (list),
   - incident_time (string),
   - incident_location (string).

4. **Priority Analysis**:  
   Rate the importance on a scale from 0 (very low) to 4 (very high).

5. **Short Title**:  
   Provide a brief title summarizing the comment (or "none").

6. **Toxicity Level**:  
   Rate toxicity (0: none to 4: very toxic).

7. **Translation**:  
   Provide an English translation (or "none").

8. **isBot**:  
   Indicate whether it is bot-generated (0: false, 1: true).

9. **Mentions/Hashtags**:  
   List any mentions or hashtags.

Return the output in the exact JSON structure:

{{
    "sentiment": <integer>,
    "intent": <integer>,
    "ner": {{
        "person_mentioned": <list>,
        "product_mentioned": <list>,
        "incident_time": "<string>",
        "incident_location": "<string>"
    }},
    "priority": <integer>,
    "short_title": "<string>",
    "toxicity_level": <integer>,
    "translation": "<string>",
    "isBot": <integer>,
    "mentions_hashtags": <list>
}}"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        result_text = response.choices[0].message.content.strip()
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON response", "raw_response": result_text}
        return result
    except Exception as e:
        return {"error": str(e)}

def flatten_analysis_result(analysis_result):
    """
    Flatten the nested NER dictionary into separate keys so that it can be displayed in a table.
    """
    if isinstance(analysis_result, dict) and "ner" in analysis_result:
        ner = analysis_result.pop("ner")
        if isinstance(ner, dict):
            analysis_result["ner_person_mentioned"] = ner.get("person_mentioned", [])
            analysis_result["ner_product_mentioned"] = ner.get("product_mentioned", [])
            analysis_result["ner_incident_time"] = ner.get("incident_time", "")
            analysis_result["ner_incident_location"] = ner.get("incident_location", "")
    return analysis_result

def analyze_comments(df_reviews, openai_api_key):
    """
    Analyze review comments concurrently with a progress bar.
    Processes 3 comments at a time.
    """
    results = []
    total = len(df_reviews)
    progress_bar = st.progress(0)
    openai_client = OpenAI(api_key=openai_api_key)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_index = {
            executor.submit(process_comment_seq, row, openai_client): (idx, row.to_dict())
            for idx, row in df_reviews.iterrows()
        }
        for i, future in enumerate(concurrent.futures.as_completed(future_to_index)):
            idx, original_data = future_to_index[future]
            try:
                analysis_result = future.result()
            except Exception as exc:
                analysis_result = {"error": str(exc)}
            analysis_result = flatten_analysis_result(analysis_result)
            combined = {**original_data, **analysis_result}
            results.append(combined)
            progress_bar.progress((i + 1) / total)
    return pd.DataFrame(results)

def display_analysis(df_analyzed, show_sentiment, show_intent, show_ner):
    """Display dashboard visualizations & insights."""
    st.markdown("## Dashboard Visualizations & Insights")
    st.write("Below you will find detailed visual explanations for each diagram derived from the review analyses.")
   
    # 1. Intent Analysis
    st.subheader("Intent Analysis")
    st.write(
        "This pie chart displays the distribution of review intent categories. "
        "Each wedge represents the combined total for categories that share the same count. "
        "The labels for wedges with the same count are shown as comma‑separated values."
    )
    if 'intent' in df_analyzed.columns:
        intent_counts = df_analyzed['intent'].value_counts().reindex(range(6), fill_value=0)
        intent_labels = ['Support', 'Criticism', 'Enquiry', 'Complaint', 'Praise', 'Sarcasm']
        grouped_vals, grouped_labels = group_pie_data(intent_counts.tolist(), intent_labels)
        fig_intent, ax_intent = plt.subplots(figsize=(5, 5))
        ax_intent.pie(
            grouped_vals,
            labels=grouped_labels,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.7,
            labeldistance=1.1,
        )
        ax_intent.set_title("Intent Distribution")
        center_pyplot(fig_intent)
   
    # 2. Sentiment Analysis
    st.subheader("Sentiment Analysis")
    st.write(
        "The following pie charts show the distribution of review sentiments. "
        "The first chart displays the raw sentiment distribution, while the aggregated chart groups sentiments "
        "into Negative, Neutral, and Positive categories. Wedges with identical values have combined, comma‑separated labels."
    )
    if 'sentiment' in df_analyzed.columns:
        sentiment_counts = df_analyzed['sentiment'].value_counts().reindex(range(5), fill_value=0)
        sentiment_labels = ['Very Angry', 'Angry', 'Neutral', 'Happy', 'Very Happy']
        grouped_vals, grouped_labels = group_pie_data(sentiment_counts.tolist(), sentiment_labels)
        fig_sent, ax_sent = plt.subplots(figsize=(5, 5))
        ax_sent.pie(
            grouped_vals,
            labels=grouped_labels,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.7,
            labeldistance=1.1,
        )
        ax_sent.set_title("Sentiment Distribution")
        center_pyplot(fig_sent)
       
        # Aggregated Sentiment
        st.write("Aggregated Sentiment:")
        def agg_sent(x):
            if x in [0, 1]:
                return "Negative"
            elif x == 2:
                return "Neutral"
            else:
                return "Positive"
        agg = df_analyzed['sentiment'].apply(agg_sent)
        agg_counts = agg.value_counts()
        grouped_vals, grouped_labels = group_pie_data(list(agg_counts.values), list(agg_counts.index))
        fig_agg = px.pie(
            names=grouped_labels,
            values=grouped_vals,
            title="Aggregated Sentiment"
        )
        fig_agg.update_layout(width=500, height=500)
        center_plotly(fig_agg)
   
    # 3. Named Entity Recognition (NER)
    st.subheader("Named Entity Recognition (NER)")
    st.write(
        "The following bar charts show the frequency of named entities extracted from the reviews, "
        "including persons mentioned, products mentioned, incident times, and locations."
    )
    ner_fields = ['ner_person_mentioned', 'ner_product_mentioned', 'ner_incident_time', 'ner_incident_location']
    available_ner = [f for f in ner_fields if f in df_analyzed.columns]
    if available_ner:
        for field in available_ner:
            items = []
            for val in df_analyzed[field].dropna():
                if isinstance(val, list):
                    items.extend(val)
            if items:
                counts = Counter(items)
                df_counts = pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).sort_values('Count', ascending=False)
                st.write(f"**Frequency for {field.replace('ner_', '').replace('_', ' ').title()}:**")
                if field in ["ner_person_mentioned", "ner_product_mentioned"]:
                    # Create a horizontal bar chart using Plotly Express
                    fig = px.bar(
                        df_counts,
                        x="Count",
                        y=df_counts.index,
                        orientation="h",
                        title=f"Frequency for {field.replace('ner_', '').replace('_', ' ').title()}"
                    )
                    fig.update_layout(yaxis=dict(title=""), xaxis_title="Count")
                    st.plotly_chart(fig)
                else:
                    st.bar_chart(df_counts)
    else:
        st.info("No NER data available.")
   
    # 4. Toxicity Analysis
    st.subheader("Toxicity Analysis")
    st.write(
        "The pie chart below represents the ratio between non‑toxic and toxic reviews as determined by the toxicity analysis. "
        "Identical values are grouped."
    )
    if 'toxicity_level' in df_analyzed.columns:
        tox_group = df_analyzed['toxicity_level'].apply(lambda x: "Non-Toxic" if x == 0 else "Toxic")
        tox_counts = tox_group.value_counts()
        grouped_vals, grouped_labels = group_pie_data(tox_counts.tolist(), list(tox_counts.index))
        fig_tox, ax_tox = plt.subplots(figsize=(5, 5))
        ax_tox.pie(
            grouped_vals,
            labels=grouped_labels,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.7,
            labeldistance=1.1,
        )
        ax_tox.set_title("Toxicity Ratio")
        center_pyplot(fig_tox)
   
    st.markdown("---")

# -------------------- STREAMLIT UI --------------------
st.set_page_config(layout="wide")
st.title("🏢 Understanding Public Sentiments For Businesses")

# Initialize session state variables
if "business_data" not in st.session_state:
    st.session_state.update({
        "business_data": None,
        "confirmed_url": None,
        "new_url_pending": None,
        "df_reviews": None,
        "df_analyzed": None,
        "max_reviews": None,
    })

# --- Sidebar: Business Search ---
st.sidebar.header("1. Business Search")
business_name = st.sidebar.text_input("Business Name", placeholder="Eg Amrik Sukhdev")
business_location = st.sidebar.text_input("Location", placeholder="Eg Murthal India")

if st.sidebar.button("Find Business"):
    st.session_state.update({
        "business_data": None,
        "confirmed_url": None,
        "new_url_pending": None,
        "df_reviews": None,
        "df_analyzed": None,
        "max_reviews": None,
    })
    with st.spinner("Searching for business... (up to 90 seconds)"):
        business_data = find_business(name=business_name, location=business_location)
    if business_data:
        st.session_state["business_data"] = business_data
        st.session_state["confirmed_url"] = business_data["url"]
    else:
        st.error("No business found. Please enter the business URL manually.")

if st.session_state["business_data"]:
    business_data = st.session_state["business_data"]
    st.subheader("🔍 Business Found:")
    st.write(f"**🏢 Name:** {business_data['title'].upper()}")
    st.write(f"**📍 Address:** {business_data['street'].upper()}, {business_data['city'].upper()}")
    st.write(f"**📂 Category:** {business_data.get('categoryName', 'N/A')}")
    st.write(f"**⭐ Rating:** {business_data['totalScore']}")
    new_business_url = st.text_input("Confirm/Edit Business URL", st.session_state.get("confirmed_url", ""))
    if new_business_url and new_business_url != st.session_state["confirmed_url"]:
        st.session_state["new_url_pending"] = new_business_url
        st.warning("🔄 URL changed! Click 'Update Business Details' to refresh.")
    if st.session_state.get("new_url_pending"):
        if st.button("Update Business Details"):
            with st.spinner("Fetching updated business details..."):
                updated_business_data = find_business(business_url=st.session_state["new_url_pending"])
            if updated_business_data:
                st.session_state["business_data"] = updated_business_data
                st.session_state["confirmed_url"] = st.session_state["new_url_pending"]
                st.session_state["new_url_pending"] = None
    if st.button("Confirm URL & Fetch Reviews"):
        st.session_state["confirmed_business"] = st.session_state["business_data"]

if "confirmed_business" in st.session_state:
    confirmed_data = st.session_state["confirmed_business"]
    with st.spinner("Updating business details..."):
        time.sleep(1)
    st.markdown("---")
    st.subheader(f"🏢 {confirmed_data['title'].upper()}")
    st.write(
        f"**📍 Address:** {confirmed_data['street'].upper()}, "
        f"{confirmed_data['city'].upper()}, {confirmed_data.get('state', '').upper()}"
    )
    st.write(f"**📂 Category:** {confirmed_data.get('categoryName', 'N/A')}")
    st.write(f"**⭐ Rating:** {confirmed_data['totalScore']}")
    st.write(f"🔗 [Business Link]({st.session_state['confirmed_url']})")
   
    max_reviews = st.number_input("Number of Reviews to Fetch", min_value=5, max_value=100, value=5, step=1, key="num_reviews")
    if st.button("Confirm Reviews Count", key="confirm_reviews"):
        st.session_state["max_reviews"] = max_reviews
   
    if st.session_state.get("max_reviews"):
        with st.spinner("Fetching reviews..."):
            df_reviews = scrape_reviews(st.session_state["confirmed_url"], max_reviews=st.session_state["max_reviews"])
        if df_reviews is not None:
            cols_to_drop = [
                'imageUrl', 'fid', 'cid', 'url', 'reviewsCount',
                'temporarilyClosed', 'permanentlyClosed', 'title', 'categories',
                'categoryName', 'countryCode', 'state', 'postalCode', 'city',
                'street', 'neighborhood', 'address', 'location', 'placeId',
                'isAdvertisement', 'translatedLanguage', 'originalLanguage', 'totalScore', 'textTranslated'
            ]
            df_reviews.drop(columns=cols_to_drop, errors='ignore', inplace=True)
            st.success("✅ Reviews fetched successfully!")
            st.session_state["df_reviews"] = df_reviews
            st.dataframe(df_reviews, width=800, height=300)
        else:
            st.error("No reviews found.")

if st.session_state.get("df_reviews") is not None:
    if st.button("Analyze Comments"):
        with st.spinner("Analyzing comments..."):
            df_analyzed = analyze_comments(st.session_state["df_reviews"], OPENAI_API_KEY)
            st.session_state["df_analyzed"] = df_analyzed
        st.success(f"✅ {len(df_analyzed)} comments analyzed successfully!")
        st.dataframe(df_analyzed, width=800, height=300)

if st.session_state.get("df_analyzed") is not None:
    st.sidebar.header("Control Panel")
    show_sentiment = st.sidebar.checkbox("Show Sentiment Analysis", value=True)
    show_intent = st.sidebar.checkbox("Show Intent Analysis", value=True)
    show_ner = st.sidebar.checkbox("Show Named Entity Recognition (NER)", value=True)
   
    display_analysis(st.session_state["df_analyzed"], show_sentiment, show_intent, show_ner)
