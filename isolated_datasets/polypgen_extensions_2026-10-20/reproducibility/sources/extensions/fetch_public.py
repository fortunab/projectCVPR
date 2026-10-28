"""Fetch a public HTTPS artifact into this new workspace, without credentials."""
import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALLOWED = {"huggingface.co", "raw.githubusercontent.com", "dl.fbaipublicfiles.com", "worksheets.codalab.org"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("destination")
    parser.add_argument("--sha256")
    parser.add_argument("--max-bytes", type=int, default=12000000000)
    args = parser.parse_args()
    url = urllib.parse.urlparse(args.url)
    if url.scheme != "https" or url.hostname not in ALLOWED:
        raise ValueError("Only explicitly allowed public HTTPS source hosts")
    dest = (ROOT / args.destination).resolve()
    if not dest.is_relative_to(ROOT):
        raise ValueError("Download must stay in extension folder")
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() or part.exists():
        raise FileExistsError("Existing destination/partial file preserved; choose explicit recovery")
    dest.parent.mkdir(parents=True, exist_ok=True)
    started, previous, received, digest = time.monotonic(), time.monotonic(), 0, hashlib.sha256()
    request = urllib.request.Request(args.url, headers={"User-Agent": "PolypGenResearch/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as stream:
        length = int(response.headers.get("Content-Length", 0))
        if length > args.max_bytes:
            raise ValueError("Source exceeds predeclared download size cap")
        while block := response.read(4 * 1024 * 1024):
            received += len(block)
            if received > args.max_bytes:
                raise ValueError("Streaming response exceeds size cap")
            stream.write(block)
            digest.update(block)
            if time.monotonic() - previous >= 10:
                print(f"Downloaded {received / 1e6:.1f} MB to {dest.name}", flush=True)
                previous = time.monotonic()
        if length and received != length:
            raise ValueError("Incomplete HTTP response")
        final = urllib.parse.urlparse(response.geturl())
    checksum = digest.hexdigest()
    if args.sha256 and checksum.lower() != args.sha256.lower():
        raise ValueError("Downloaded artifact does not match pinned SHA-256")
    part.replace(dest)
    meta = {"source_url": args.url, "final_host_path": f"{final.scheme}://{final.netloc}{final.path}",
            "bytes": received, "sha256": checksum, "seconds": time.monotonic() - started,
            "credentials": "No user credentials or authorization headers supplied"}
    dest.with_suffix(dest.suffix + ".source.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
