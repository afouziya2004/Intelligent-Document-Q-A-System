from loader import load_document, split_text
from embedder import create_embeddings
from retriever import retrieve

def run_qa_system():
    print("🔄 Loading document...")

    text = load_document("data/sample.txt")
    chunks = split_text(text)

    vectorizer, vectors = create_embeddings(chunks)

    print("✅ System Ready! Ask questions (type 'exit' to stop)\n")

    while True:
        query = input("❓ Your Question: ")

        if query.lower() == "exit":
            print("👋 Exiting system...")
            break

        from answer_generator import generate_answer

result, score = retrieve(query, vectorizer, vectors, chunks)

answer = generate_answer(query, result)

print("\n💡 Answer:")
print(answer)
        print(f"\n🔎 Similarity Score: {score:.4f}\n")


if __name__ == "__main__":
    run_qa_system()
