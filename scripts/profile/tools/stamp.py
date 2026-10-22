#!/usr/bin/env python3
"""Prefix every line of stdin with the unix time it arrived (the trace clock's origin is unix too)."""
import sys
import time

while True:
    line = sys.stdin.readline()
    if not line:
        break
    sys.stdout.write(f"{time.time():.3f} {line}")
    sys.stdout.flush()
