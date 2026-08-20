# =========================
# IMPORTS
# =========================
from src.data_preprocessing import load_data, preprocess_data
from src.scoring_model import calculate_score
from src.recommender import recommend_places
from src.similarity_model import build_similarity, get_similar_places


# =========================
# LOAD & PREPROCESS DATA
# =========================
df = load_data()
df = preprocess_data(df)


# =========================
# PERSONALIZED SCORING
# =========================
df = calculate_score(
    df,
    rating_weight=0.6,
    review_weight=0.2,
    significance_weight=0.1,
    budget_weight=0.1
)


# =========================
# TOP RECOMMENDED PLACES
# =========================
print("\nTop Recommended Places:")
top_places = recommend_places(df, top_n=5)
print(top_places[["name", "city", "final_score"]])


# =========================
# BUILD SIMILARITY MODEL
# =========================
similarity_matrix = build_similarity(df)


# =========================
# FIND SIMILAR PLACES
# =========================
print("\nPlaces Similar To Taj Mahal:")
similar_places = get_similar_places(
    df,
    similarity_matrix,
    place_name="taj mahal",
    top_n=5
)

print(similar_places)

