import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
DB_PATH = "data/feedback.db"
def get_db_connection():
    """Returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
def init_db():
    """Initializes the feedback database table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            predicted_prob REAL NOT NULL,
            detected_tactics TEXT NOT NULL,
            language TEXT NOT NULL,
            script TEXT NOT NULL,
            user_correction TEXT DEFAULT NULL,
            source TEXT DEFAULT 'web',
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    
    # Migration: Add source column if it doesn't exist (for existing tables)
    cursor.execute("PRAGMA table_info(analysis_logs)")
    columns = [row["name"] for row in cursor.fetchall()]
    if "source" not in columns:
        cursor.execute("ALTER TABLE analysis_logs ADD COLUMN source TEXT DEFAULT 'web'")
        conn.commit()
        
    conn.close()
    logging.getLogger("scamshield.db").info("database ready", extra={"path": DB_PATH})
def log_prediction(
    msg_id: str, 
    text: str, 
    label: str, 
    prob: float, 
    tactics: List[str], 
    language: str, 
    script: str,
    source: str = "web"
):
    """Logs a prediction transaction into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO analysis_logs (
            id, text, predicted_label, predicted_prob, detected_tactics, language, script, source, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        msg_id, 
        text, 
        label, 
        float(prob), 
        json.dumps(tactics), 
        language.lower(), 
        script.lower(), 
        source.lower(),
        timestamp
    ))
    conn.commit()
    conn.close()
def save_feedback(msg_id: str, correction: str) -> int:
    """Records a user's correction; returns the number of rows updated (0 if unknown)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE analysis_logs
        SET user_correction = ?
        WHERE id = ?
    """, (correction.lower(), msg_id))
    updated = cursor.rowcount
    conn.commit()
    conn.close()
    return updated
def get_stats() -> Dict[str, Any]:
    """Queries and returns aggregated stats from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # 1. Total count
    cursor.execute("SELECT COUNT(*) as count FROM analysis_logs")
    total_count = cursor.fetchone()["count"]
    # 2. Predicted label distribution
    cursor.execute("SELECT predicted_label, COUNT(*) as count FROM analysis_logs GROUP BY predicted_label")
    label_dist = {row["predicted_label"]: row["count"] for row in cursor.fetchall()}
    # 3. Language distribution
    cursor.execute("SELECT language, COUNT(*) as count FROM analysis_logs GROUP BY language")
    lang_dist = {row["language"].title(): row["count"] for row in cursor.fetchall()}
    # 4. Script distribution
    cursor.execute("SELECT script, COUNT(*) as count FROM analysis_logs GROUP BY script")
    script_dist = {row["script"].title(): row["count"] for row in cursor.fetchall()}
    # 5. Tactic distribution
    cursor.execute("SELECT detected_tactics FROM analysis_logs WHERE predicted_label = 'scam'")
    tactic_counts: Dict[str, int] = {}
    rows = cursor.fetchall()
    for row in rows:
        tactics = json.loads(row["detected_tactics"])
        for tactic in tactics:
            t_display = tactic.replace("_", " ").title()
            tactic_counts[t_display] = tactic_counts.get(t_display, 0) + 1
    # 6. Feedback corrections count
    cursor.execute("SELECT COUNT(*) as count FROM analysis_logs WHERE user_correction IS NOT NULL")
    feedback_count = cursor.fetchone()["count"]
    # 7. Source breakdown (Web vs. SMS)
    cursor.execute("SELECT source, COUNT(*) as count FROM analysis_logs GROUP BY source")
    source_dist = {row["source"].lower(): row["count"] for row in cursor.fetchall()}
    source_dist = {
        "web": source_dist.get("web", 0),
        "sms": source_dist.get("sms", 0)
    }
    conn.close()
    return {
        "total_analyzed": total_count,
        "label_distribution": label_dist,
        "language_distribution": lang_dist,
        "script_distribution": script_dist,
        "tactic_distribution": tactic_counts,
        "feedback_received": feedback_count,
        "source_distribution": source_dist
    }
if __name__ == "__main__":
    init_db()
    # Simple test
    log_prediction(
        "test-uuid-123", 
        "Pay within 10 minutes", 
        "scam", 
        0.95, 
        ["urgency"], 
        "English", 
        "Latin"
    )
    print(get_stats())
