#!/bin/bash
cd "$(dirname "$0")"
uvicorn server:app --reload --port 8000 &
