import streamlit as st
st.set_page_config(page_title="Vibe Matcher", page_icon="🎧", layout="centered")

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1 Load Model

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()



# 2 Mock Product Data

data = [
    {
        "name": "Boho Dress",
        "desc": "Flowy, earthy tones for festival vibes",
        "vibes": ["boho", "festival"],
    },
    {
        "name": "Urban Jacket",
        "desc": "Sleek black jacket for city nightlife",
        "vibes": ["urban", "chic"],
    },
    {
        "name": "Sport Sneakers",
        "desc": "Comfortable sneakers for energetic movement",
        "vibes": ["sporty", "energetic"],
    },
    {
        "name": "Classic Blazer",
        "desc": "Sharp tailored blazer for professional events",
        "vibes": ["formal", "classic"],
    },
    {
        "name": "Cozy Sweater",
        "desc": "Soft wool sweater for relaxing cozy days",
        "vibes": ["cozy", "casual"],
    },
]

df = pd.DataFrame(data)



# 3 Precompute embeddings

def embed_products(df):
    df = df.copy()
    df["embedding"] = df["desc"].apply(
        lambda x: model.encode([x], convert_to_numpy=True)[0]
    )
    return df

df = embed_products(df)



# 4 Vibe Matching Function

def match_products(query, top_k=3):
    q_emb = model.encode([query], convert_to_numpy=True)

    sims = cosine_similarity(q_emb, np.vstack(df["embedding"].values))[0]

    df_tmp = df.copy()
    df_tmp["similarity"] = sims
    df_tmp = df_tmp.sort_values("similarity", ascending=False)

    return df_tmp.head(top_k)



# 5 Streamlit UI

st.title("🎧 Vibe-Matcher")
st.subheader("Find outfits matching your vibe")

query = st.text_input("Describe your vibe (ex: energetic urban chic):")

if st.button("Match"):
    if not query.strip():
        st.warning("Enter a vibe")
    else:
        results = match_products(query)

        if results.empty:
            st.error("⚠️ No match found! Try a different vibe.")
        else:
            st.success("Here are your top matches ↓")

            for _, row in results.iterrows():
                st.markdown(
                    f"""
                    ### ✅ {row['name']}
                    **Description:** {row['desc']}  
                    **Tags:** {', '.join(row['vibes'])}  
                    **Similarity Score:** `{row['similarity']:.4f}`
                    """
                )


# Footer
st.markdown("---")
st.caption("Nexora Vibe Matcher Prototype")
