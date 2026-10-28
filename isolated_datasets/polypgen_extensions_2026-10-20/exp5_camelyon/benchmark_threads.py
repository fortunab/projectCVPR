"""Resource-only frozen encoder benchmark; never reads labels or model scores."""
import sys,time,json,hashlib
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision.models import resnet18,ResNet18_Weights
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
paths=sorted((ROOT/"images").glob("patient*/*.png"))[:64]
weights_path=Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
model=resnet18(weights=None)
model.load_state_dict(torch.load(weights_path,map_location="cpu",weights_only=True))
model.fc=torch.nn.Identity()
model.requires_grad_(False).eval()
transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
torch.set_num_threads(2)
with threadpool_limits(limits=2):
    images=[]
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    pixels=torch.stack(images)
results=[]
with torch.inference_mode():
    for threads in [2,4]:
        torch.set_num_threads(threads)
        with threadpool_limits(limits=threads):
            model(pixels)
            timings=[]
            for repetition in range(3):
                tick=time.perf_counter()
                outputs=model(pixels).numpy()
                timings.append(time.perf_counter()-tick)
            results.append({"threads":threads,"batch_size":len(paths),"timed_repetitions_seconds":timings,
                            "median_seconds":float(np.median(timings)),"images_per_second":len(paths)/float(np.median(timings))})
            print(json.dumps(results[-1]),flush=True)
result={"benchmark":"fixed64imagebatch;forwardonly;warmup1;timed3;torch2vs4",
        "labels_read":False,"scores_read":False,"results":results,
        "concurrent_jobs":"DINOv2 8threads and original ResNet extraction2threads",
        "image_names_sha256":hashlib.sha256("\n".join(p.name for p in paths).encode()).hexdigest()}
(ROOT/"features/thread_benchmark.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
