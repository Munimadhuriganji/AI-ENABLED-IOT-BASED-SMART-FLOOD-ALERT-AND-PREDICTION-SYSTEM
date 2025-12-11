import sqlite3, os, csv, json
DB_PATH = os.environ.get("DATABASE", "./data/floodwatch.db")

def export_alerts_csv(path="data/alerts_export.csv"):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,node_id,risk,prob,reason,ts FROM alerts ORDER BY ts DESC")
    rows = c.fetchall()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","node","risk","prob","reason","ts"])
        w.writerows(rows)
    conn.close()
    print("Exported", len(rows), "alerts to", path)

if __name__ == "__main__":
    export_alerts_csv()
