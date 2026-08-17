import argparse
import json
import statistics
from pathlib import Path

def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None

def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    args = parser.parse_args()

    path = Path(args.jsonl_path)

    with path.open("r", encoding = "utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        raise ValueError("No results found in JSON file")

    avg_runtime = mean([
        row.get("generation_runtime_s")
        for row in rows
    ])

    avg_ttft = mean([
        row.get("ttft_s")
        for row in rows
    ])


    total_itl_time = 0.0
    total_itl_count = 0

    for row in rows:
        mean_itl = row.get("mean_itl_s")
        itl_count = row.get("itl_count", 0)

        if mean_itl is not None and itl_count > 0:
            total_itl_time += mean_itl * itl_count
            total_itl_count += itl_count


    avg_itl = (
        total_itl_time / total_itl_count
        if total_itl_count > 0
        else None
    )

    avg_generated_tokens = mean([
        row.get("generated_tokens")
        for row in rows
    ])

    avg_kv_tokens = mean([
        row.get("kv_tokens")
        for row in rows
    ])

    avg_full_kv_tokens = mean([
        row.get("full_equivalent_kv_tokens")
        for row in rows
    ])

    avg_kv_memory_mb = mean([
        row.get("kv_memory_mb")
        for row in rows
    ])

    total_kv_tokens = sum(
        row.get("kv_tokens", 0)
        for row in rows
        if row.get("kv_tokens") is not None
    )

    total_full_kv_tokens = sum(
        row.get("full_equivalent_kv_tokens", 0)
        for row in rows
        if row.get("full_equivalent_kv_tokens") is not None
    )

    total_generated_tokens = sum(
        row.get("generated_tokens", 0)
        for row in rows
    )
    

    if total_full_kv_tokens > 0:
        kv_retention = total_kv_tokens / total_full_kv_tokens
        kv_reduction = 1.0 - kv_retention
    else:
        kv_retention = None
        kv_reduction = None

    median_runtime = median([
        row.get("generation_runtime_s")
        for row in rows
    ])

    median_ttft = median([
        row.get("ttft_s")
        for row in rows
    ])


    print(f"Samples: {len(rows)}")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Avg generated tokens: {avg_generated_tokens:.2f}")

    print(f"Avg generation runtime: {avg_runtime:.4f} s")
    print(f"Median generation runtime: {median_runtime:.4f} s")

    print(f"Avg TTFT: {avg_ttft:.4f} s")
    print(f"Median TTFT: {median_ttft:.4f} s")

    if avg_itl is not None:
        print(f"Mean ITL: {avg_itl:.4f} s")
        print(f"Mean ITL: {avg_itl * 1000:.2f} ms")
    
    print(f"Avg retained KV layer-token slots: {avg_kv_tokens:.2f}")
    print(f"Avg full-equivalent KV slots: {avg_full_kv_tokens:.2f}")
    print(f"Avg KV memory: {avg_kv_memory_mb:.2f} MB")

    if kv_retention is not None:
        print(f"KV retention: {kv_retention * 100:.2f}%")
        print(f"KV reduction: {kv_reduction * 100:.2f}%")
    

if __name__ == "__main__":
    main()

    