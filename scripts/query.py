"""Quick event browser for data/counts.db.

Usage:
    python scripts/query.py              # last 20 events
    python scripts/query.py -n 50        # last 50 events
    python scripts/query.py -n 0         # all events
    python scripts/query.py --hours 6    # last 6 hours
    python scripts/query.py --today      # today only
    python scripts/query.py --summary    # hourly + source breakdown, no row listing
"""

import argparse
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "counts.db"


def parse_args():
    p = argparse.ArgumentParser(description="Query oak-car-counter event log.")
    p.add_argument("-n", type=int, default=20, metavar="N",
                   help="Show last N events (0 = all, default 20)")
    p.add_argument("--hours", type=float, default=None,
                   help="Limit to last N hours")
    p.add_argument("--today", action="store_true",
                   help="Limit to today (UTC)")
    p.add_argument("--summary", action="store_true",
                   help="Print summary stats only, skip event listing")
    return p.parse_args()


def main():
    args = parse_args()

    if not DB.exists():
        print(f"Database not found: {DB}")
        return

    conn = sqlite3.connect(str(DB))

    # Build WHERE clause
    filters = []
    if args.today:
        filters.append("date(timestamp) = date('now')")
    elif args.hours is not None:
        filters.append(f"timestamp >= datetime('now', '-{args.hours} hours')")
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    # --- Summary ---
    total = conn.execute(f"SELECT COUNT(*) FROM events {where}").fetchone()[0]
    with_speed = conn.execute(
        f"SELECT COUNT(*) FROM events {where} {'AND' if where else 'WHERE'} speed_kmh IS NOT NULL"
    ).fetchone()[0]

    print(f"\nTotal events : {total}")
    print(f"With speed   : {with_speed}  ({100*with_speed//total if total else 0}%)")

    print("\nBy source:")
    for src, cnt in conn.execute(
        f"SELECT COALESCE(source,'?'), COUNT(*) FROM events {where} GROUP BY source ORDER BY COUNT(*) DESC"
    ):
        print(f"  {src:<25} {cnt}")

    print("\nBy class:")
    for cls, cnt, avg in conn.execute(
        f"SELECT vehicle_class, COUNT(*), ROUND(AVG(speed_kmh),1) FROM events {where}"
        f" GROUP BY vehicle_class ORDER BY COUNT(*) DESC"
    ):
        avg_str = f"{avg} km/h avg" if avg is not None else "no speed"
        print(f"  {cls:<12} {cnt:>4}   {avg_str}")

    print("\nHourly (UTC):")
    for hour, cnt in conn.execute(
        f"SELECT strftime('%H:00',timestamp), COUNT(*) FROM events {where}"
        f" GROUP BY 1 ORDER BY 1"
    ):
        bar = "#" * cnt
        print(f"  {hour}  {cnt:>3}  {bar}")

    if args.summary:
        conn.close()
        return

    # --- Event listing ---
    limit = f"LIMIT {args.n}" if args.n > 0 else ""
    rows = conn.execute(
        f"SELECT timestamp, vehicle_class, direction, speed_kmh, source, frame_path"
        f" FROM events {where} ORDER BY id DESC {limit}"
    ).fetchall()
    conn.close()

    if not rows:
        print("\nNo events found.")
        return

    print(f"\nLast {len(rows)} event(s):\n")
    print(f"{'Timestamp':<32} {'Class':<10} {'Dir':<6} {'Speed':>8}  {'Source':<25} Frame")
    print("-" * 100)
    for ts, cls, direction, speed, source, fp in reversed(rows):
        speed_str = f"{speed:>7.1f}" if speed is not None else "   None"
        frame_str = "yes" if fp else "-"
        print(f"{ts:<32} {cls:<10} {direction:<6} {speed_str}  {str(source):<25} {frame_str}")


if __name__ == "__main__":
    main()
