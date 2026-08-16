from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from src.data_preprocessing import load_data, preprocess_data
from src.scoring_model import calculate_score
from src.recommender import recommend_places
from src.similarity_model import build_similarity, get_similar_places


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AI Travel Recommender",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI Travel Recommendation System")
st.markdown("#### Personalized + ML-Based Hybrid Travel Engine")

st.markdown("---")

# -------------------------
# Load Data
# -------------------------
df = load_data()
df = preprocess_data(df)

# -------------------------
# Sidebar - Personalization
# -------------------------
st.sidebar.header("⚙ Personalization Settings")

rating_weight = st.sidebar.slider("⭐ Rating Importance", 0.0, 1.0, 0.4)
review_weight = st.sidebar.slider("📈 Popularity Importance", 0.0, 1.0, 0.3)
significance_weight = st.sidebar.slider("🏛 Significance Importance", 0.0, 1.0, 0.2)
budget_weight = st.sidebar.slider("💰 Budget Importance", 0.0, 1.0, 0.1)

# Normalize weights
total = rating_weight + review_weight + significance_weight + budget_weight
if total == 0:
    total = 1

rating_weight /= total
review_weight /= total
significance_weight /= total
budget_weight /= total

df = calculate_score(
    df,
    rating_weight=rating_weight,
    review_weight=review_weight,
    significance_weight=significance_weight,
    budget_weight=budget_weight
)

# -------------------------
# Filters
# -------------------------
st.subheader("🔍 Explore Places")

col1, col2 = st.columns(2)

with col1:
    place_type = st.selectbox("Select Type", ["All"] + sorted(df["type"].unique().tolist()))

with col2:
    best_time = st.selectbox("Best Time to Visit", ["All"] + sorted(df["best_time_to_visit"].unique().tolist()))

filtered_df = df.copy()

if place_type != "All":
    filtered_df = filtered_df[filtered_df["type"] == place_type]

if best_time != "All":
    filtered_df = filtered_df[
        filtered_df["best_time_to_visit"].str.contains(best_time)
    ]

st.markdown("---")

# -------------------------
# Top Recommendations
# -------------------------
st.subheader("🏆 Top Recommended Places")

top_places = filtered_df.sort_values(by="final_score", ascending=False).head(6)

cols = st.columns(3)

for index, row in top_places.iterrows():
    with cols[index % 3]:
        st.markdown(f"### {row['name'].title()}")
        st.markdown(f"📍 {row['city'].title()}")
        st.markdown(f"⭐ Rating: {row['google_review_rating']}")
        st.markdown(f"🎟 Entry Fee: ₹{row['entrance_fee_in_inr']}")
        st.progress(float(row["final_score"]))
        st.markdown("---")

# -------------------------
# Similar Places Section
# -------------------------
st.subheader("🔁 Find Similar Places")

similarity_matrix = build_similarity(df)

place_name_input = st.selectbox("Choose a Place", sorted(df["name"].unique().tolist()))

if st.button("Find Similar Places"):
    similar_places = get_similar_places(
        df,
        similarity_matrix,
        place_name=place_name_input,
        top_n=5
    )

    st.markdown("### 🔎 Similar Places")

    sim_cols = st.columns(5)

    for i, (_, row) in enumerate(similar_places.iterrows()):
        with sim_cols[i]:
            st.markdown(f"**{row['name'].title()}**")
            st.markdown(f"📍 {row['city'].title()}")
            st.markdown(f"🏷 {row['type'].title()}")

@st.cache_data
def get_coordinates(df):
    geolocator = Nominatim(user_agent="travel_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    df = df.copy()
    df["location"] = df["city"].apply(geocode)
    df["latitude"] = df["location"].apply(lambda loc: loc.latitude if loc else None)
    df["longitude"] = df["location"].apply(lambda loc: loc.longitude if loc else None)

    df = df.dropna(subset=["latitude", "longitude"])

    return df


st.markdown("---")
st.subheader("🗺 Map View of Recommended Places")

if not top_places.empty:

    map_df = get_coordinates(top_places)

    st.map(
        map_df,
        latitude="latitude",
        longitude="longitude"
    )
else:
    st.write("No places to display on map.")


