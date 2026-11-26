"""Per-request artifacts, bounded lifetime and traversal-safe downloads."""
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
ARTIFACTS = frozenset({"mask.png", "colored.png", "overlay.png", "result.json"} | {
    f"slic-{name}" for name in ("labels.npy", "regions.png", "boundary.png")
} | {f"cam-{class_id}.png" for class_id in range(1, 81)})


class ResultStore:
    def __init__(self, root, ttl):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.lock = threading.RLock()

    def create(self):
        request_id = uuid.uuid4().hex
        directory = self.root / request_id
        directory.mkdir()
        return request_id, directory

    def publish(self, request_id, result):
        result["request_id"] = request_id
        result["expires_at"] = time.time() + self.ttl
        result["storage_format"] = "segmentation-service-v1"
        base = f"/api/v1/results/{request_id}/"
        result["outputs"] = {key + "_url": base + name for key, name in result["outputs"].items()}
        for stage in result["stages"].values():
            if "artifacts" in stage:
                stage["artifacts"] = {key + "_url": base + name
                                      for key, name in stage["artifacts"].items()}
        directory = self.root / request_id
        with self.lock:
            temporary = directory / "result.tmp"
            temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            temporary.replace(directory / "result.json")
        return result

    def discard(self, request_id):
        if not REQUEST_ID.fullmatch(request_id):
            raise ValueError("invalid request ID")
        with self.lock:
            directory = self.root / request_id
            if directory.is_dir() and not directory.is_symlink():
                shutil.rmtree(directory)

    def read(self, request_id, artifact):
        if not REQUEST_ID.fullmatch(request_id) or artifact not in ARTIFACTS:
            raise FileNotFoundError
        with self.lock:
            directory = self.root / request_id
            if directory.is_symlink():
                raise FileNotFoundError
            manifest = directory / "result.json"
            path = directory / artifact
            if manifest.is_symlink() or path.is_symlink():
                raise FileNotFoundError
            result = json.loads(manifest.read_text())
            if result["expires_at"] <= time.time():
                raise FileNotFoundError
            allowed = {url.rsplit("/", 1)[-1] for url in result["outputs"].values()}
            for stage in result["stages"].values():
                allowed.update(url.rsplit("/", 1)[-1] for url in stage.get("artifacts", {}).values())
            if artifact != "result.json" and artifact not in allowed:
                raise FileNotFoundError
            return path.read_bytes()

    def cleanup(self):
        with self.lock:
            for directory in self.root.iterdir():
                if not REQUEST_ID.fullmatch(directory.name) or directory.is_symlink() or not directory.is_dir():
                    continue
                manifest = directory / "result.json"
                if manifest.is_symlink():
                    continue
                try:
                    result = json.loads(manifest.read_text())
                    if (result.get("storage_format") == "segmentation-service-v1"
                            and result["expires_at"] <= time.time()):
                        shutil.rmtree(directory)
                except (OSError, ValueError, KeyError):
                    continue
