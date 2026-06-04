import requests
import sys

API_URL = "http://localhost:8000"

if len(sys.argv) < 2:
    print("Usage: python api_test_client.py 'your question here'")
    sys.exit(1)

query = sys.argv[1]
resp = requests.post(f"{API_URL.rstrip('/')}/query", json={"query": query, "include_context": True, "top_k": 3})
resp.raise_for_status()
print("Answer:\n")
print(resp.json().get("answer"))
print('\nRetrieved docs:')
for c in resp.json().get("context", []):
    print(f"- {c.get('source')} (chunk {c.get('chunk_index')}): {c.get('text')[:200]}...")
