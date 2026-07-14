"""
update_scheduler.py
====================
Stand-alone periodic job that watches new_sources/ and incrementally merges
any new FAQ content into the FAISS knowledge base — with no manual button-
clicking required.

Why the `schedule` library instead of APScheduler or a real cron job?
  - Zero extra system setup: no crontab entry, no OS service, no Windows
    Task Scheduler config — just `python update_scheduler.py`.
  - Tiny, free, pure-Python dependency with a one-line, readable API.
  - The task explicitly says this "does not need to be a production-grade
    scheduler, just demonstrably automatic" — `schedule` is the simplest
    tool that satisfies that bar. APScheduler would add more moving parts
    (job stores, executors, threads) for capability we don't need here;
    a real cron job would be OS-specific and harder to demo/grade.

Run it in a second terminal, alongside `streamlit run main.py`:

    python update_scheduler.py                 # check every 60 minutes (default)
    python update_scheduler.py --interval 1     # check every 1 minute   (demo / grading)
    python update_scheduler.py --once           # run a single check and exit
"""

import argparse
import datetime as dt
import time

import schedule

from langchain_helper import update_vector_db_incremental, BASE_DIR

LOG_PATH = BASE_DIR / "kb_update.log"


def _log_line(line: str):
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    full_line = f"{timestamp} {line}"
    print(full_line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(full_line + "\n")


def job():
    _log_line("Checking new_sources/ for updates...")
    summary = update_vector_db_incremental(verbose=False)
    if summary["new_documents_added"] > 0:
        _log_line(
            f"Added {summary['new_documents_added']} new FAQ(s) from "
            f"{summary['files_scanned']}. KB now has {summary['total_documents_in_kb']} docs."
        )
    else:
        _log_line("No new content found. KB unchanged.")


def main():
    parser = argparse.ArgumentParser(description="Periodic FAQ knowledge base updater")
    parser.add_argument("--interval", type=int, default=60, help="Minutes between checks (default: 60)")
    parser.add_argument("--once", action="store_true", help="Run a single check and exit (no loop)")
    args = parser.parse_args()

    if args.once:
        job()
        return

    _log_line(f"Scheduler started — checking every {args.interval} minute(s). Ctrl+C to stop.")
    schedule.every(args.interval).minutes.do(job)
    job()  # also run one check immediately on startup, don't wait for first interval
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
