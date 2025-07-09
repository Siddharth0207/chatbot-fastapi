from fastapi.testclient import TestClient
from app import app
import concurrent.futures
import random
import string
import json
import sys
import asyncio
import os

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def random_session_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

def send_query(client, query, session_id):
    headers = {"x-session-id": session_id}
    payload = {"query": query}
    response = client.post("/query", json=payload, headers=headers)
    return response

def test_concurrent_queries():
    client = TestClient(app)
    queries = [
        "Find a 3 carat diamonds with clarity VS1",
        "Show me 2 carat diamonds",
        "Diamonds from India with EX cut and 2 carat weight",
        "Cheap diamonds with 3 carat weight",
        "Best value diamonds 2 carat ",
        "45 carat Diamonds with strong fluorescence",
        "Round shape diamonds with 2.5 carat ",
        "2 carat Diamonds with price under 5000",
        "4 carat Diamonds with VVS1 clarity",
        "3 carat Diamonds with GIA lab"
    ]
    num_requests = 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(num_requests):
            query = random.choice(queries)
            session_id = random_session_id()
            futures.append(executor.submit(send_query, client, query, session_id))
        results = [f.result() for f in futures]
    # Check all responses and collect for saving
    output = []
    for idx, resp in enumerate(results):
        assert resp.status_code == 200
        assert "x-session-id" in resp.headers
        assert isinstance(resp.json(), dict)
        output.append({
            "index": idx+1,
            "session_id": resp.headers["x-session-id"],
            "response": resp.json()
        })
    print(f"Current working directory: {os.getcwd()}")
    try:
        with open("concurrency_test_output.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print("File concurrency_test_output.json written successfully.")
    except Exception as e:
        print(f"Error writing file: {e}")
    print(f"{num_requests} concurrent requests completed successfully. Responses saved to concurrency_test_output.json.")

if __name__ == "__main__":
    test_concurrent_queries()
