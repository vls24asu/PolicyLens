"""Shared pytest configuration — sets stub env vars so Settings() doesn't fail
when ANTHROPIC_API_KEY is not present in the test environment.
"""

import os

# Provide stub values for any required Settings fields before any module is imported.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-stub-for-unit-tests")
os.environ.setdefault("NEO4J_PASSWORD",    "test-password")
