import json
from vector_db import QuadrantStorage, COLLECTION_NAME

def inspect_data():
    """Simple script to print data stored in Qdrant."""
    db = QuadrantStorage()
    print(f"\n[Qdrant] Inspecting collection: '{COLLECTION_NAME}'")
    
    # Scroll through points (retrieves points without needing a query vector)
    result = db.client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    
    points = result[0]
    if not points:
        print("❌ Database is currently empty.")
        return

    print(f"✅ Found {len(points)} points sample (Total may be higher).")
    print("-" * 60)
    
    for p in points:
        source = p.payload.get("source", "Unknown")
        idx = p.payload.get("chunk_index", "N/A")
        text_snippet = p.payload.get("text", "")[:100].replace("\n", " ")
        print(f"📄 Source: {source} | Chunk: {idx}")
        print(f"📝 Text: {text_snippet}...")
        print("-" * 60)

if __name__ == "__main__":
    try:
        inspect_data()
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
