#!/usr/bin/env bash
set -e
python -m src.data
python -m src.train
python -m src.evaluate
