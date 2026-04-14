import faiss
import numpy as np

# Create FAISS index
def create_faiss_index(vectors):
    dense_vectors = vectors.toarray().astype('float32')

    dimension = dense_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(dense_vectors)

    return index, dense_vectors


# Search using FAISS
def search_faiss(query_vec, index, k=1):
    query_vec = query_vec.toarray().astype('float32')

    distances, indices = index.search(query_vec, k)

    return indices[0]
