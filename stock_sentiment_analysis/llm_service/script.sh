#!/bin/bash
cd "$(dirname "$0")"
uvicorn llm_app:app --reload --host 127.0.0.1 --port 8001 &  # <-- note the &
