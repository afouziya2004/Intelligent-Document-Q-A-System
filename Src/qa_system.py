from loader import load_document, split_text
from embedder import create_embeddings
from retriever import retrieve
from faiss_index import create_faiss_index, search_faiss

def run_qa_system():
    print("🔄 Loading document...")

    text = load_document("data/sample.txt")
    chunks = split_text(text)

    vectorizer, vectors = create_embeddings(chunks)
    index, dense_vectors = create_faiss_index(vectors)

    print("✅ System Ready! Ask questions (type 'exit' to stop)\n")

    while True:
        query = input("❓ Your Question: ")

        if query.lower() == "exit":
            print("👋 Exiting system...")
            break

        from answer_generator import generate_answer

query_vec = vectorizer.transform([query])
indices = search_faiss(query_vec, index)

result = chunks[indices[0]]

answer = generate_answer(query, result)

print("\n💡 Answer:")
print(answer)
        print(f"\n🔎 Similarity Score: {score:.4f}\n")


if __name__ == "__main__":
    run_qa_system()
