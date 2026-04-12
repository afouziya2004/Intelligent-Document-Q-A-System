from sklearn.metrics.pairwise import cosine_similarity

# Retrieve most relevant chunk
def retrieve(query, vectorizer, vectors, chunks):
    query_vec = vectorizer.transform([query])

    similarity = cosine_similarity(query_vec, vectors).flatten()

    best_index = similarity.argmax()

    return chunks[best_index], similarity[best_index]


# Test retrieval system
if __name__ == "__main__":
    from loader import load_document, split_text
    from embedder import create_embeddings

    text = load_document("data/sample.txt")
    chunks = split_text(text)

    vectorizer, vectors = create_embeddings(chunks)

    query = "What is AI used for?"

    result, score = retrieve(query, vectorizer, vectors, chunks)

    print("Query:", query)
    print("\nBest Match:\n", result)
    print("\nSimilarity Score:", score)
