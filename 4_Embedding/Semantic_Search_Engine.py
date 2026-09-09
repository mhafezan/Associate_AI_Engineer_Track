"""Search Shopify products using OpenAI embeddings, without RAG.

Install with your chosen interpreter: python -m pip install numpy openai tiktoken
Set OPENAI_API_KEY, then run this file. Use --limit to change the catalogue size.
Dataset: https://huggingface.co/datasets/Shopify/product-catalogue (Apache-2.0)
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from openai import OpenAI, OpenAIError
import tiktoken


MODEL = "text-embedding-3-small"
CACHE_DIR = Path(__file__).resolve().parent / ".semantic_search_cache"
client = None


def load_products(limit=1000):
    """Download product dictionaries in pages, reusing a local JSON cache."""
    if limit < 5:
        raise ValueError("The product limit must be at least 5.")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"shopify_train_{limit}_v1.json"
    if cache_file.exists():
        try:
            products = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(products, list) and len(products) >= 5 and all(
                isinstance(product, dict)
                and all(key in product for key in
                        ("title", "short_description", "category", "features"))
                for product in products
            ):
                return products
        except (ValueError, TypeError):
            pass

    products = []
    offset = 0
    while len(products) < limit:
        params = urlencode({
            "dataset": "Shopify/product-catalogue", "config": "default",
            "split": "train", "offset": offset, "length": min(100, limit - len(products)),
        })
        with urlopen(f"https://datasets-server.huggingface.co/rows?{params}", timeout=60) as response:
            page = json.load(response)
        rows = page["rows"]
        if not rows:
            break
        for item in rows:
            row = item["row"]
            title = (row.get("product_title") or "").strip()
            if not title:
                continue
            brand = (row.get("ground_truth_brand") or "").strip()
            products.append({
                "title": title,
                # Keep descriptions short for display and embedding costs.
                "short_description": (row.get("product_description") or "").strip()[:2000],
                "category": row.get("ground_truth_category") or "Uncategorized",
                "features": [f"Brand: {brand}"] if brand else [],
            })
        offset += len(rows)
        print(f"Loaded {len(products):,} products...", flush=True)
        if offset >= page["num_rows_total"]:
            break
    if len(products) < 5:
        raise ValueError("The dataset returned fewer than five usable products.")
    cache_file.write_text(json.dumps(products, ensure_ascii=False), encoding="utf-8")
    return products


def create_product_text(product):
    """Combine the product fields into one searchable string."""
    return (
        f"Title: {product['title']}\n"
        f"Description: {product['short_description']}\n"
        f"Category: {product['category']}\n"
        f"Features: {'; '.join(product['features'])}"
    )


def create_embeddings(texts):
    """Embed a string or list of strings in bounded batches, preserving order."""
    global client
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    encoding = tiktoken.encoding_for_model(MODEL)
    inputs = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding inputs must be nonempty strings.")
        tokens = encoding.encode(text, disallowed_special=())
        if len(tokens) > 8191:
            raise ValueError("An embedding input exceeds 8,191 tokens; shorten it.")
        inputs.append(tokens)
    if client is None:
        client = OpenAI(timeout=60, max_retries=3)

    embeddings = []
    start = 0
    while start < len(inputs):
        end, token_count = start, 0
        # Bound both item count and total tokens per request.
        while end < len(inputs) and end - start < 64:
            if token_count + len(inputs[end]) > 24000:
                break
            token_count += len(inputs[end])
            end += 1
        response = client.embeddings.create(
            model=MODEL, input=inputs[start:end], encoding_format="float",
        )
        data = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in data] != list(range(end - start)):
            raise ValueError("The API returned incomplete or misindexed embeddings.")
        embeddings.extend(item.embedding for item in data)
        start = end
        if len(inputs) > 1:
            print(f"Embedded {start:,}/{len(inputs):,} products...", flush=True)
    return embeddings


def find_n_closest(query_vector, embeddings, n=5):
    """Return product indices and cosine distances in ascending order."""
    if n <= 0 or len(embeddings) == 0:
        return []
    embeddings = np.asarray(embeddings, dtype=np.float32)
    query_vector = np.asarray(query_vector, dtype=np.float32)
    if embeddings.ndim != 2 or query_vector.shape != (embeddings.shape[1],):
        raise ValueError("Query and product embedding dimensions must match.")
    if not np.isfinite(embeddings).all() or not np.isfinite(query_vector).all():
        raise ValueError("Embeddings must contain only finite values.")
    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0 or np.any(norms == 0):
        raise ValueError("Cosine distance is undefined for zero vectors.")
    distances = 1 - np.clip((embeddings @ query_vector) / (norms * query_norm), -1, 1)
    distances_sorted = np.argsort(distances, kind="stable")[:n]
    return [{"distance": float(distances[index]), "index": int(index)}
            for index in distances_sorted]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000, help="Products to load (default: 1000)")
    args = parser.parse_args()
    if args.limit < 5:
        parser.error("--limit must be at least 5")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError("Set the OPENAI_API_KEY environment variable before running.")

    products = load_products(args.limit)
    product_texts = [create_product_text(product) for product in products]
    # Invalidate vectors whenever the model, text, or product order changes.
    fingerprint = hashlib.sha256(
        json.dumps([MODEL, product_texts], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    cache_file = CACHE_DIR / f"embeddings_{fingerprint}.npy"
    product_embeddings = None
    if cache_file.exists():
        try:
            cached = np.load(cache_file, allow_pickle=False)
            if (cached.shape == (len(products), 1536) and np.isfinite(cached).all()
                    and np.all(np.linalg.norm(cached, axis=1) > 0)):
                product_embeddings = cached
                print("Using cached product embeddings.")
        except (ValueError, EOFError):
            pass
    if product_embeddings is None:
        product_embeddings = np.asarray(create_embeddings(product_texts), dtype=np.float32)
        np.save(cache_file, product_embeddings, allow_pickle=False)

    print(f"Ready to search {len(products):,} products. Type exit to quit.")
    while True:
        try:
            query_text = input("\nSearch: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if query_text.casefold() == "exit":
            print("Goodbye!")
            break
        if not query_text:
            continue
        try:
            query_vector = create_embeddings(query_text)[0]
            hits = find_n_closest(query_vector, product_embeddings, 5)
        except (OpenAIError, ValueError) as error:
            print(f"Search failed: {error}", file=sys.stderr)
            continue
        print(f'\nSearch results for "{query_text}"')
        for rank, hit in enumerate(hits, start=1):
            product = products[hit["index"]]
            print(f"\n{rank}. {product['title']} (cosine distance: {hit['distance']:.4f})")
            print(f"   Category: {product['category']}")
            print(f"   {' '.join(product['short_description'].split())}")
            if product["features"]:
                print(f"   {'; '.join(product['features'])}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except (OpenAIError, URLError, OSError, ValueError, KeyError) as error:
        print(f"Unable to run semantic search: {error}", file=sys.stderr)
        sys.exit(1)
