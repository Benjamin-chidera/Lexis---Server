"""
conftest.py

This file is loaded by pytest BEFORE any test module is imported.

The problem it solves:
  database.py raises a ValueError at module level if DATABASE_URL is
  not set. When pytest collects tests, it imports test files, which
  import main.py, which imports database.py — causing the crash before
  any test fixture can run.

The fix:
  We set a fake DATABASE_URL here so database.py passes its guard check.
  The actual tests use an in-memory SQLite database (via fixture overrides),
  so this dummy URL is never used for real connections.
"""

import os

# Set a dummy DATABASE_URL so database.py does not crash at import time.
# Individual test files override the actual DB engine with an in-memory
# SQLite database, so this URL is never used to make a real connection.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_placeholder.db")
