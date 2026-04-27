# SQLite Cases Database – Prerequisites and Setup

The Support Assistant can **list** and **create** support cases from a local SQLite database. This document covers prerequisites, setup steps, and loading mock data.

---

## 1. Prerequisites

- **Python 3.9+** (same as the main project).
- **SQLite3** – included in Python’s standard library; no extra install.
- **Project dependencies** – install as usual:
  ```bash
  pip install -r requirements.txt
  ```
- **Write access** to the project **data directory** (default: `data/` under the project root). The DB file will be created there.

---

## 2. Configuration

The DB file path is set via environment or default:

| Variable        | Default           | Description                    |
|----------------|-------------------|--------------------------------|
| `SQLITE_DB_PATH` | `data/cases.db`   | Full path to the SQLite file.  |

Example in `.env`:

```env
# Optional: use a custom path (default: data/cases.db)
SQLITE_DB_PATH=data/cases.db
```

The table is created automatically on first use (list or create). No manual DB creation is required.

---

## 3. Steps to Create and Use the Database

### Step 1: Ensure the data directory exists

From the project root:

```bash
mkdir -p data
```

On Windows:

```cmd
mkdir data
```

(If you use the default path, the app also creates it on startup.)

### Step 2: (Optional) Load mock data

To prefill the DB with sample cases for testing and demos:

```bash
# From project root
python -m database.seed_mock_data
```

This will:

- Create the `cases` table if it does not exist.
- Insert **6 mock cases** (4 Active, 2 Closed) if the table is empty.

To insert mock data even when the table already has rows:

```bash
python -m database.seed_mock_data --force
```

### Step 3: Run the API

Start the FastAPI app as usual:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- **List cases (API):** `GET /cases` or `GET /cases?status=Active`
- **Create case (API):** `POST /cases` with body `{"title": "My issue", "description": "Optional details", "status": "Active"}`

### Step 4: Use from the chat UI

In the RAG chat:

- **List open/active cases:** e.g.  
  - “Get me all open cases”  
  - “List active cases”  
  - “Show all active open cases”
- **Create a case:** e.g.  
  - “I’d like to create a case”  
  - “Create a case”  
  - “Create a case about login problem”

The assistant will either list cases from the DB or create a new Active case and confirm.

---

## 4. Mock Data Loaded by `seed_mock_data`

The seed script inserts the following rows (status and text only; IDs and timestamps are generated):

| Title                              | Description                                                                 | Status  |
|------------------------------------|-----------------------------------------------------------------------------|---------|
| Login failure for dealer portal    | User cannot log in with valid credentials; getting 401 after password reset. | Active  |
| Invoice export missing line items  | Exported CSV for January 2026 is missing line items for orders after 15th.   | Active  |
| Password reset email not received  | Dealer requested password reset three times; no email received.              | Active  |
| Order status stuck at Pending      | Order #ORD-2026-0042 has been Pending for 48 hours. Payment was confirmed. | Active  |
| Duplicate charges on bulk order    | Bulk order was charged twice. Refund requested for second charge.            | Closed  |
| Catalog PDF download 404           | Link to Q4 2025 catalog returns 404. Resolved by updating link in portal.   | Closed  |

After seeding, “Get me all open cases” / “List active cases” will return the 4 Active cases; “Create a case” will add a new row with status `Active`.

---

## 5. Schema Reference

Table: **cases**

| Column       | Type    | Description                    |
|-------------|---------|--------------------------------|
| id          | INTEGER | Primary key (auto)            |
| title       | TEXT    | Required                       |
| description | TEXT    | Optional                       |
| status      | TEXT    | Default `'Active'` (e.g. Active, Closed) |
| created_at  | TEXT    | ISO 8601 UTC timestamp         |

---

## 6. Troubleshooting

- **“No such table: cases”** – The table is created on first use. Ensure the app or `python -m database.seed_mock_data` has run at least once and that `SQLITE_DB_PATH` (or the default `data/` directory) is writable.
- **“There are no active open cases”** – Either no rows exist or none have `status = 'Active'`. Run `python -m database.seed_mock_data` to load mock data.
- **Chat does not list/create cases** – Confirm the server is running and the chat is calling the same backend. List/create intents are detected by phrases like “all open cases” and “create a case”; rephrase if needed.
