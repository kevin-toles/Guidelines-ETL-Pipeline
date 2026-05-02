#!/usr/bin/env python3
"""
Streaming tag graph + summary generator for StackExchange output.
Reads existing primary/hacks/supplemental JSONL files and streams:
  - tag_graph.csv (for Neo4j import) — written row-by-row, no memory accumulation
  - pipeline_summary.json — computed incrementally from counters

Usage:
  python3 generate_tag_graph_summary.py \
    /Volumes/USB321FD/Guidelines\ ETL\ Data/ai-platform-output/stackexchange
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import gc


def stream_tag_graph(output_dir):
    """Stream-write tag graph CSV and return incremental per-site stats.
    
    Returns (tag_edge_count, per_site_stats) where per_site_stats is a dict
    of site -> {collection: doc_count, ...}.
    """
    tag_path = output_dir / 'tag_graph.csv'
    per_site = defaultdict(lambda: defaultdict(int))
    tag_edge_count = 0
    batch = []
    batch_size = 5000  # write every N tag rows
    
    with open(tag_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tag', 'question_id', 'site', 'tag_count'])
        
        for col_name in ['primary', 'hacks', 'supplemental']:
            col_dir = output_dir / col_name
            if not col_dir.exists():
                print(f"  SKIP: {col_name}/ — directory not found")
                continue
            
            jsonl_files = sorted([p for p in col_dir.glob('*.jsonl') 
                                  if not p.name.startswith('._')])
            total_files = len(jsonl_files)
            
            for fi, jsonl_file in enumerate(jsonl_files):
                site = jsonl_file.stem
                doc_count = 0
                
                with open(jsonl_file) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            doc = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        doc_count += 1
                        
                        qid = doc.get('id', '')
                        tags = doc.get('tags', [])
                        if tags and qid:
                            tc = len(tags)
                            for tag in tags:
                                batch.append([tag, qid, site, tc])
                                if len(batch) >= batch_size:
                                    writer.writerows(batch)
                                    tag_edge_count += len(batch)
                                    batch.clear()
                
                per_site[site][col_name] = doc_count
                
                if total_files > 1 and (fi + 1) % 20 == 0:
                    print(f"    [{fi+1}/{total_files}] {col_name}: {tag_edge_count:,} edges so far, gc...")
                    gc.collect()
            
            # flush remaining for this collection
            if batch:
                writer.writerows(batch)
                tag_edge_count += len(batch)
                batch.clear()
            gc.collect()
    
    return tag_edge_count, per_site


def write_summary(output_dir, tag_edge_count, per_site):
    """Write pipeline_summary.json from incremental stats."""
    site_list = []
    total_per_col = defaultdict(int)
    
    for site in sorted(per_site.keys()):
        site_doc_count = sum(per_site[site].values())
        site_list.append({
            'site': site,
            'docs': site_doc_count,
            'primary': per_site[site].get('primary', 0),
            'hacks': per_site[site].get('hacks', 0),
            'supplemental': per_site[site].get('supplemental', 0),
        })
        for col in ['primary', 'hacks', 'supplemental']:
            total_per_col[col] += per_site[site].get(col, 0)
    
    total_docs = sum(total_per_col.values())
    summary = {
        'run_time': datetime.now(timezone.utc).isoformat(),
        'total_sites': len(site_list),
        'tag_edges': tag_edge_count,
        'totals': {
            'all': total_docs,
            **{col: total_per_col[col] for col in ['primary', 'hacks', 'supplemental']}
        },
        'per_site': site_list
    }
    
    summary_path = output_dir / 'pipeline_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return summary_path, total_docs


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    print(f"Reading from: {output_dir}")
    
    tag_edge_count, per_site = stream_tag_graph(output_dir)
    
    print(f"\n{'='*60}")
    print(f"TAG GRAPH: {tag_edge_count:,} tag→question edges across {len(per_site)} sites")
    
    summary_path, total_docs = write_summary(output_dir, tag_edge_count, per_site)
    print(f"DOCS:     {total_docs:,} documents (primary + hacks + supplemental)")
    print(f"SUMMARY:  {summary_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
