#!/bin/bash
git add .
git commit -m "Backup: $(date)"
git push
source ./venv/bin/activate 