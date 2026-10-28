"""Resource-only CPU benchmark; no fitting or label-dependent operations."""
import time
import torch
from threadpoolctl import threadpool_limits
from exp4_dinov2 import (OUT, batch_pixels, dump_json, load_cohort,
                         load_encoder, normalized_cls)

rows, _, _ = load_cohort()
torch.set_num_threads(4)
encoder, processor = load_encoder()
pixels = batch_pixels(rows[:32], processor)
results = []
for threads, batch in ((4, 16), (8, 16), (4, 32), (8, 32)):
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        normalized_cls(encoder, pixels[:batch])
        tick = time.perf_counter()
        normalized_cls(encoder, pixels[:batch])
        elapsed = time.perf_counter() - tick
    record = {"threads": threads, "batch_size": batch, "timed_seconds": elapsed,
              "images_per_second": batch / elapsed, "images": len(rows),
              "estimated_full_forward_seconds": elapsed * len(rows) / batch}
    results.append(record)
    print(record, flush=True)
fastest = max(results, key=lambda r: r["images_per_second"])
if fastest["images_per_second"] < results[0]["images_per_second"] * 1.05:
    fastest = results[0]
dump_json(OUT / "extraction_resource_plan.json", {
    "benchmark": results, "selected_threads": fastest["threads"],
    "selected_batch_size": fastest["batch_size"], "head_threads": 4,
    "policy": "Fixed first32 manifest images, same pinned preprocessing/checkpoint; warmup then one measured forward per resource cell; select throughput,preferoriginal4threads/batch16ifimprovementbelow5percent; no labels used, head fitting fixed4threads.",
    "workers": 0, "feature_precision": "float32"})
print("SELECTED", fastest, flush=True)
