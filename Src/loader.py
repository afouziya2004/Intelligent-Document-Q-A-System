# Load full document
def load_document(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return text


# Split document into chunks
def split_text(text, chunk_size=20):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


# Test (important for today)
if __name__ == "__main__":
    text = load_document("data/sample.txt")
    chunks = split_text(text)

    print("Total Chunks:", len(chunks))
    print("\nSample Chunk:\n", chunks[0])
