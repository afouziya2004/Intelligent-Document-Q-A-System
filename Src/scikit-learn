from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text chunks into embeddings (vectors)
def create_embeddings(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors


# Test embedding pipeline
if __name__ == "__main__":
    from loader import load_document, split_text

    text = load_document("data/sample.txt")
    chunks = split_text(text)

    vectorizer, vectors = create_embeddings(chunks)

    print("Number of chunks:", len(chunks))
    print("Vector shape:", vectors.shape)
