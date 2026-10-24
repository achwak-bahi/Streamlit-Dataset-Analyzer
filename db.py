"""Stockage SQLite des resultats d'analyse (JSON).

Plan :  Dataset -> tp.py -> JSON -> SQLite (analysis.db) -> Dashboard
"""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "analysis.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS analyses (
               dataset_name TEXT PRIMARY KEY,
               result_json  TEXT NOT NULL,
               analyzed_at  TEXT NOT NULL
           )"""
    )
    return conn


def save_result(dataset_name, result):
    """Enregistre (ou remplace) le JSON d'analyse d'un dataset."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = json.dumps(result, ensure_ascii=False)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO analyses (dataset_name, result_json, analyzed_at) VALUES (?, ?, ?)",
            (dataset_name, payload, now),
        )


def get_result(dataset_name):
    """Retourne le JSON (dict) d'un dataset, ou None s'il n'est pas encore en base."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT result_json FROM analyses WHERE dataset_name = ?", (dataset_name,)
        ).fetchone()
    return json.loads(row[0]) if row else None
