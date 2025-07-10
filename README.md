# Diamond Search Chatbot

A modern FastAPI-based chatbot for searching diamonds using natural language queries. The system leverages an LLM (NVIDIA/Meta) for robust preference extraction, normalizes user input, queries a PostgreSQL database asynchronously, and returns both a summary and a list of matching diamonds. The frontend features a modern, responsive chat UI with a typing effect and multi-turn conversation support.

## Features

- **Natural Language Search:** Users can search for diamonds using plain English queries.
- **LLM-Powered Extraction:** Uses an LLM to extract and normalize diamond preferences from user input.
- **Async PostgreSQL Queries:** Fast, concurrent database access using SQLAlchemy async and asyncpg.
- **Multi-Turn Chat & Session Support:** Maintains chat history per user session.
- **Modern Frontend:** Responsive chat UI with typing/streaming effect and professional design.
- **Total Count Reporting:** Returns both the top results and the total number of matches.
- **Logging:** All events and errors are logged to `logs/diamond_app.log`.
- **Concurrency Tested:** Includes tests for concurrent and load scenarios.

## Project Structure

```
├── app.py                # FastAPI backend with endpoints
├── main.py               # DiamondFinder logic, LLM, normalization, SQL
├── db.py                 # (Optional) DB utilities
├── home.html             # Frontend chat UI
├── requirements.txt      # Python dependencies
├── logger.py             # Logging setup (file + console)
├── test_app.py           # Basic endpoint tests
├── test_concurrency.py   # Concurrency/load test
├── logs/
│   └── diamond_app.log   # Log file (auto-created)
└── ...
```

## Setup & Run

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up PostgreSQL

- Ensure PostgreSQL is running and accessible.
- Update the DB URL in `app.py` and `main.py` if needed (default: `postgresql+asyncpg://postgres:0207@localhost:5432/postgres`).
- Import your diamonds data into a table named `diamonds`.

### 3. Run the FastAPI backend

```bash
uvicorn app:app --reload
```

### 4. Open the Frontend

- Open `home.html` in your browser (or serve it via a static server).
- The frontend will connect to the backend at `http://localhost:8000` (update if needed).

## Testing

- Run basic API tests:

  ```
    bash
  pytest test_app.py
  ```

- Run concurrency/load test (and see output in `concurrency_test_output.json`):
  ```
    bash
  python test_concurrency.py
  ```

## Logging

- All logs (info, errors) are saved to `logs/diamond_app.log` and printed to the console.

## Customization

- **LLM Model:** Change the model in `main.py` if you want to use a different LLM.
- **Frontend:** Edit `home.html` for UI/UX changes.
- **Normalization:** Update mappings in `main.py` for new diamond properties or value variations.

## License

Apache License 2.0

---

**Contact:** For questions or support, open an issue or contact the maintainer.
