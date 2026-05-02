#!/bin/bash
cd /Users/kevintoles/POC/ai-platform-data
python3 scripts/etl/generate_tag_graph_summary.py \
  "/Volumes/USB321FD/Guidelines ETL Data/ai-platform-output"
echo "Done. Press Cmd+W to close."
read -p "Press Enter to exit..."
