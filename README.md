# Vibe Matcher — Mini Recommender Prototype

This notebook implements a lightweight "Vibe Matcher" that takes a natural-language vibe description
(e.g., "energetic urban chic") and retrieves the top-3 matching fashion items using semantic embeddings.

It simulates a minimal recommendation workflow without requiring external services.

---

## Features

✔ Mock product dataset (5 items, each with name/description/vibes)  
✔ Create embeddings using `sentence-transformers/all-MiniLM-L6-v2`  
(OpenAI free tier not applicable for me)   
✔ Query to encode to cosine similarity ranking  
✔ Returns top-N items with similarity scores  
✔ Basic evaluation (score > 0.7 → “good match”)  
✔ Latency measurement with `time.time()`  
✔ Reflection + future improvements included
  

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language  | Python |
| Notebook  | Jupyter |
| Embeddings | Sentence-Transformers (MiniLM-L6-v2) |
| Similarity | Cosine similarity |
| Eval | Manual threshold |
| Visualization | matplotlib |

---

## Workflow

1) Load mock product dataset  
2) Generate text embeddings for products  
3) Convert user query to embedding  
4) Compute cosine similarity  
5) Rank items to return top-3  
6) Evaluate 3 test vibe queries  
7) Plot inference latency  

---

## Example Query

query = "energetic urban chic"


Expected top match:
- Sporty / Urban items (e.g., Leather Jacket / Tracksuit)

---

## Results

- Similarity scores computed correctly
- Multiple queries tested
- Matches mostly aligned with product vibes

### Performance
Avg latency per query ≈ **10–30ms** on CPU  
Good enough for prototype scale.

---

## Streamlit Web App

A simple UI to interact with the vibe-matcher.

## 📸 Web App Screenshot

Below is a preview of the Streamlit interface:

<img src="screenshots/ui.PNG" alt="UI" width="600">



---

## Possible Improvements

- Switch embeddings to OpenAI or Cohere
- Store vectors in Pinecone / FAISS for real-time retrieval
- Add images → multimodal matching
- Learnable re-ranking via LLM
- De-duplicate via metadata filters

---

# Final Submission Checklist

- Notebook runs top-to-bottom without errors  
- Dataset → Embeddings → Matching → Evaluation → Latency → Reflection  
- Code modular + readable  
- README present  
- No hard-coded API keys  
- No unnecessary files  




