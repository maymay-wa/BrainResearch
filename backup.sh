#!/bin/bash
source ./venv/bin/activate 
git add .
git commit -m "Backup: $(date)"
git push