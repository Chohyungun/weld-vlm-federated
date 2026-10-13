"""Read-only, loopback-only experiment observatory. Python standard library only."""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import statistics
import math
import tarfile
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "prep_v3_imgonly_20261008T2335"
IDENTITY = "06ac84c187639a06e82e1f0542abe337431f6cc091a1dd9d229dc1f71b4b621e"
ARCHIVE_SHA = "a79e737050fea3699075a3a591285fe594828b18b95290751b1d1b6ff1ab650b"
TASKS_SHA = "7c290d018d893379313d359d9f1b0a7e8993a7b6e5a7f1ff08462632829bd1c1"
MANIFEST_SHA = "6847045f2a2bf5d17ac0aa7e99769da035e365e92e642347ed18b67bc82595e5"
RUN_SHA = "51a94da91d199f08851a1cf4e348e0ed0a1ff5c90bc4ca29d28119a3b5e3d1d1"
RECOVERY = "20261009T0452_budget_stop"
PREFIX = "20261008T194804818474Z_00023_recovery"
UTC = timezone.utc


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stamp():
    return datetime.now(UTC).isoformat()


def number(value, default=None):
    return value if type(value) in (int, float) and math.isfinite(value) and abs(value) < 1e12 else default


def short(value, maximum=240):
    return str(value or "")[:maximum]


def event_view(event):
    """Do not pass through raw command lines, machine paths, credentials or logs."""
    counts = event.get("counts", event.get("snapshot", {})) or {}
    report = event.get("report", {}) or {}
    messages = {
        "reviewed_resume_verified": "기존 완료 자료 검증 후 판정 재개",
        "projection_report_only": "남은 판정 시간·비용 추정 (부분 수행)",
        "call_started": "새 호출 시작", "call_finished": "호출 종료",
        "progress_verified": "완료 샤드 검증", "completed_prefix_frozen": "완료 파일 무결성 확인",
        "recovery_verified": "회수 묶음 저장·해시 검증", "recovery_started": "자료 회수 시작",
        "stopped": "작업 중단", "projection": "전량 시간·비용 예상",
    }
    return {
        "at": short(event.get("at", event.get("observed_at_utc")), 50),
        "event": short(event.get("event"), 70),
        "stage": short(event.get("stage"), 20),
        "reason": short(event.get("reason"), 100) if re.fullmatch(r"[a-zA-Z0-9_ :.-]*", str(event.get("reason", ""))) else "details_redacted",
        "generated_shards": number(counts.get("generate")),
        "judged_shards": number(counts.get("judge", event.get("shards"))),
        "message": messages.get(event.get("event"), ""),
        "external_rc": number(event.get("outer_rc", event.get("external_rc", event.get("rc")))),
        "cost_estimate_won": number(event.get("cost_estimate_won", event.get("cost_won"))),
        "projected_won": number(report.get("projected_won")),
    }


def judge_view(row):
    if not row:
        return {"status": "pending", "text": None, "reason": "로컬 회수 묶음에 판정 결과가 없습니다."}
    try:
        parsed = json.loads(row["text"])
        keys = ("grounded", "scope_ok", "language_ok")
        if not isinstance(parsed, dict) or not all(type(parsed.get(k)) is bool for k in keys):
            raise ValueError("schema")
        return {"status": "all_true" if all(parsed[k] for k in keys) else "flagged", **{k: parsed[k] for k in keys}, "reason": short(parsed.get("reason"), 5000), "text": row["text"], "finish_reason": row.get("finish_reason")}
    except (ValueError, TypeError, KeyError):
        return {"status": "parse_error", "reason": "판정 JSON 파싱/스키마 확인 실패 — 검토 필요", "text": row.get("text"), "finish_reason": row.get("finish_reason")}


class Observatory:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.run_dir = self.root / "outputs/corpus_production" / RUN_ID
        self.recovery_dir = self.run_dir / "server_return" / RECOVERY
        self.live_path = self.root / "dashboard/runtime/live_status.json"
        self.tasks = {}
        self.generated = {}
        self.judged = {}
        self.shards = []
        self.events = []
        self.image_hash_cache = {}
        self.error = None
        self.loaded_at = stamp()
        try:
            self._load()
        except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
            self.tasks.clear()
            self.generated.clear()
            self.judged.clear()
            self.error = type(exc).__name__ + ": 원천 자료 확인 실패. 터미널의 데이터 경로와 SHA를 확인하세요."

    def _load(self):
        run_path, task_path = self.run_dir / "RUN.json", self.run_dir / "tasks.jsonl"
        manifest_path = self.recovery_dir / (PREFIX + ".manifest.json")
        archive_path = self.recovery_dir / (PREFIX + ".tar")
        for path, expected in ((run_path, RUN_SHA), (task_path, TASKS_SHA), (manifest_path, MANIFEST_SHA), (archive_path, ARCHIVE_SHA)):
            if sha_file(path) != expected:
                raise ValueError("source hash mismatch")
        self.run = read_json(run_path)
        if self.run["identity_sha256"] != IDENTITY:
            raise ValueError("run identity")
        manifest = read_json(manifest_path)
        with task_path.open(encoding="utf-8") as f:
            for line in f:
                task = json.loads(line)
                if task.get("split") not in ("train", "val") or task.get("kind") != "image_report":
                    raise ValueError("non train/val image task")
                self.tasks[task["id"]] = {k: task.get(k) for k in ("id", "client", "split", "kind", "image_path", "image_sha256", "evidence", "locked_target")}
        if len(self.tasks) != self.run["counts"]["tasks"]:
            raise ValueError("task count")
        with tarfile.open(archive_path, "r:") as archive:
            # Never extract archives. Read regular, manifest-verified members only.
            for member in archive:
                stage = "generate" if re.fullmatch(r"run/generated/\d{6}\.json", member.name) else "judge" if re.fullmatch(r"run/judged/\d{6}\.json", member.name) else None
                if not stage:
                    continue
                if not member.isfile() or member.size > 5_000_000:
                    raise ValueError("unsafe archive member")
                payload = archive.extractfile(member).read()
                if hashlib.sha256(payload).hexdigest() != manifest.get(member.name):
                    raise ValueError("shard hash")
                shard = json.loads(payload)
                if shard["identity_sha256"] != IDENTITY:
                    raise ValueError("shard identity")
                destination = self.generated if stage == "generate" else self.judged
                for row in shard["rows"]:
                    if row["id"] not in self.tasks or row["id"] in destination:
                        raise ValueError("shard task identity")
                    destination[row["id"]] = row
                self.shards.append({"stage": stage, "index": shard["index"], "at": shard["created_at"], "rows": len(shard["rows"]), **{k: shard["metrics"].get(k) for k in ("wall_seconds", "output_tokens", "peak_reserved_bytes", "model_load_seconds")}})
        ledger_path = self.recovery_dir / "ledger.jsonl"
        if ledger_path.exists():
            self.events = [event_view(json.loads(line)) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.status = read_json(self.root / "_workspace/2026-10-08-overnight/STATUS.json")
        self.splits, self.clients, self.classes, self.instances = Counter(), Counter(), Counter(), Counter()
        self.judge_counts = Counter()
        self.normal = 0
        for task in self.tasks.values():
            self.splits[task["split"]] += 1
            self.clients[task["client"]] += 1
            defects = task["evidence"].get("defect_counts_by_name", {})
            if not defects:
                self.normal += 1
            for name, count in defects.items():
                self.classes[name] += 1
                self.instances[name] += count
        for row in self.judged.values():
            self.judge_counts[judge_view(row)["status"]] += 1
        self.search_index = {tid: " ".join([tid, task["client"], *task["evidence"].get("defect_counts_by_name", {}), self.generated.get(tid, {}).get("text", "")]).lower() for tid, task in self.tasks.items()}
        self.ordered_ids = list(self.tasks)

    def live(self):
        result = {"available": False, "stale": True, "source": "local_recovery", "observed_at_utc": None}
        if not self.live_path.exists():
            return result
        try:
            if self.live_path.stat().st_size > 1_000_000:
                raise ValueError("oversized live status")
            raw = read_json(self.live_path)
            if raw.get("run_id") != RUN_ID or raw.get("identity") != IDENTITY:
                raise ValueError("wrong run")
            observed = datetime.fromisoformat(raw["observed_at_utc"].replace("Z", "+00:00"))
            age = (datetime.now(UTC) - observed).total_seconds()
            if observed.tzinfo is None or age < -30:
                raise ValueError("invalid live timestamp")
            values = {}
            for key, ceiling in (("generated_shards", 779), ("judged_shards", 779), ("generated_rows", 49845), ("judged_rows", 49845), ("pending_judge_rows", 49845), ("cost_estimate_won", 1000000), ("cap_won", 100000)):
                value = number(raw.get(key))
                if value is not None and (value < 0 or value > ceiling):
                    raise ValueError("invalid live count")
                if key.endswith(("_shards", "_rows")) and value is not None and type(value) is not int:
                    raise ValueError("non integer count")
                values[key] = value
            for stage in ("generated", "judged"):
                count, rows = values[stage + "_shards"], values[stage + "_rows"]
                if count is not None and rows is not None and rows != min(count * 64, 49845):
                    raise ValueError("row/shard mismatch")
            if values["judged_rows"] is not None and values["pending_judge_rows"] is not None and values["judged_rows"] + values["pending_judge_rows"] != 49845:
                raise ValueError("pending count mismatch")
            gpu = raw.get("gpu") or {}
            if not isinstance(gpu, dict) or not isinstance(raw.get("events", []), list):
                raise ValueError("invalid live structure")
            events = [event_view(x) for x in raw.get("events", [])[-100:] if isinstance(x, dict)]
            result.update(available=True, stale=age > 120, age_seconds=max(0, round(age)), source="server-readback", observed_at_utc=observed.isoformat(), **values)
            result.update(phase=short(raw.get("phase"), 100), supervisor_running=raw.get("supervisor_running") is True, hard_stop_at_utc=short(raw.get("hard_stop_at_utc"), 50), gpu={k: number(gpu.get(k)) for k in ("utilization_pct", "memory_used_mb")}, events=events)
            validated = raw.get("validated_counts", {})
            if isinstance(validated, dict):
                count = number(validated.get("judged_shards"))
                if count is not None and 0 <= count <= (values["judged_shards"] or 0):
                    result["validated_judged_shards"] = count
            error_path = self.live_path.with_name("live_status.error.json")
            if error_path.is_file():
                poll_error = read_json(error_path)
                result["poll_error"] = poll_error.get("active") is True
                if result["poll_error"]:
                    result["stale"] = True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            result = {"available": False, "stale": True, "source": "local_recovery", "observed_at_utc": None}
            result["error"] = "라이브 스냅샷을 읽지 못했습니다. 회수 자료를 표시합니다."
        return result

    def summary(self):
        if self.error:
            return {"error": self.error, "loaded_at": self.loaded_at}
        live = self.live()
        historical_cost = self.status.get("final_conservative_cost_upper_estimate_won")
        models = self.run["config"]
        return {
            "run_id": RUN_ID, "identity": IDENTITY, "code": "0f798df", "loaded_at": self.loaded_at,
            "snapshot_at": self.status["observed_at_kst"], "live": live,
            "tasks": len(self.tasks), "total_shards": (len(self.tasks) + 63) // 64,
            "generated_rows": len(self.generated), "judged_rows": len(self.judged),
            "generated_shards": sum(s["stage"] == "generate" for s in self.shards),
            "judged_shards": sum(s["stage"] == "judge" for s in self.shards),
            "cost_estimate_won": historical_cost, "cost_as_of": self.status.get("instance_stop_confirmed_no_later_than_kst"), "cap_won": 100000,
            "splits": dict(self.splits), "clients": dict(self.clients), "classes": dict(self.classes), "defect_instances": dict(self.instances),
            "normal_rows": self.normal, "defect_rows": len(self.tasks) - self.normal,
            "source_qa": self.run["counts"]["source_qa"], "judge_counts": dict(self.judge_counts),
            "models": {stage: {k: models[stage][k] for k in ("model_id", "revision", "quantization")} for stage in ("generation", "judge", "training")},
            "stage_metrics": {stage: self.metrics(stage) for stage in ("generate", "judge")},
            "training": {"status": "not_started", "metrics": [], "note": "이 run은 학습 릴리스 전입니다. 현재 연결된 run의 학습 지표는 없습니다. 테스트/모의 지표를 실험 성능으로 표시하지 않습니다."},
            "evidence": [{"name": "RUN.json + tasks.jsonl", "status": "SHA-256 확인", "scope": "고정 train/val 명부"}, {"name": PREFIX + ".tar", "status": "SHA-256 확인", "scope": "779 생성 · 89 판정 샤드"}, {"name": "ledger.jsonl + STATUS.json", "status": "로컬 회수 기록", "scope": "2026-10-09 종료 시점"}, {"name": "live_status.json", "status": "신규 스냅샷" if live["available"] else "미연결", "scope": "120초 이후 stale"}],
        }

    def metrics(self, stage):
        shards = [s for s in self.shards if s["stage"] == stage]
        durations = [s["wall_seconds"] for s in shards if number(s.get("wall_seconds")) is not None]
        return {"median_shard_seconds": round(statistics.median(durations), 2) if durations else None, "output_tokens": sum(s.get("output_tokens") or 0 for s in shards), "peak_reserved_gb": round(max((s.get("peak_reserved_bytes") or 0 for s in shards), default=0) / 1024**3, 2), "series": [{"x": s["index"], "y": s["wall_seconds"]} for s in shards]}

    def samples(self, query):
        if self.error:
            return {"total": 0, "offset": 0, "limit": 12, "items": [], "error": self.error}
        term = query.get("q", [""])[0].lower()[:200]
        split, client, defect, judgment = [query.get(k, [""])[0] for k in ("split", "client", "defect", "judge")]
        offset = max(0, int(query.get("offset", [0])[0]))
        limit = min(50, max(1, int(query.get("limit", [12])[0])))
        found = []
        for tid in self.ordered_ids:
            task = self.tasks[tid]
            if split and task["split"] != split or client and task["client"] != client:
                continue
            defects = task["evidence"].get("defect_counts_by_name", {})
            if defect == "normal" and defects or defect == "defect" and not defects:
                continue
            if term and term not in self.search_index[tid]:
                continue
            if judgment and judge_view(self.judged.get(tid))["status"] != judgment:
                continue
            found.append(tid)
        return {"total": len(found), "offset": offset, "limit": limit, "items": [self.sample(tid, False) for tid in found[offset:offset + limit]]}

    def sample(self, tid, detailed=True):
        task = self.tasks[tid]
        generated = self.generated.get(tid, {})
        judged = judge_view(self.judged.get(tid))
        result = {"id": tid, "split": task["split"], "client": task["client"], "defects": task["evidence"].get("defect_counts_by_name", {}), "text": generated.get("text"), "finish_reason": generated.get("finish_reason"), "judge": judged if detailed else {"status": judged["status"]}, "image_available": self.image_path(tid, verify=False) is not None}
        if detailed:
            result.update(evidence=task["evidence"], boxes=task["locked_target"].get("defects", []), image_sha256=task["image_sha256"], verdict=task["locked_target"].get("verdict"), output_tokens=generated.get("output_tokens"))
        return result

    def image_path(self, tid, verify=True):
        task = self.tasks.get(tid)
        if not task or task.get("split") not in ("train", "val"):
            return None
        relative = Path(task.get("image_path", ""))
        if relative.is_absolute() or ".." in relative.parts or any(p.lower() in {"evaluation", "eval", "test", ".git", ".ssh"} for p in relative.parts):
            return None
        path = (self.root / relative).resolve()
        data_root = (self.root / "data").resolve()
        if not path.is_relative_to(data_root) or not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            return None
        if path.stat().st_size > 30_000_000:
            return None
        if verify:
            stat = path.stat()
            signature = (str(path), stat.st_mtime_ns, stat.st_size)
            if signature not in self.image_hash_cache:
                self.image_hash_cache[signature] = sha_file(path)
            if self.image_hash_cache[signature] != task["image_sha256"]:
                return None
        return path


class Handler(BaseHTTPRequestHandler):
    store: Observatory
    static_root = Path(__file__).resolve().parent / "static"

    def log_message(self, fmt, *args):
        # Avoid echoing search text or dataset identifiers into terminal logs.
        pass

    def send(self, payload, content_type="application/json; charset=utf-8", status=200):
        if not isinstance(payload, bytes):
            payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        host = self.headers.get("Host", "").split(":")[0].lower()
        if host not in ("127.0.0.1", "localhost"):
            return self.send({"error": "loopback host only"}, status=403)
        target = urlsplit(self.path)
        query = parse_qs(target.query)
        try:
            if target.path == "/api/summary":
                return self.send(self.store.summary())
            if target.path == "/api/events":
                live = self.store.live()
                return self.send({"items": self.store.events + live.get("events", []), "live": live["available"] and not live["stale"]})
            if target.path == "/api/samples":
                return self.send(self.store.samples(query))
            if target.path == "/api/sample":
                return self.send(self.store.sample(query.get("id", [""])[0]))
            if target.path == "/api/image":
                path = self.store.image_path(query.get("id", [""])[0])
                if path is None:
                    return self.send({"error": "whitelisted train/val image unavailable"}, status=404)
                return self.send(path.read_bytes(), mimetypes.guess_type(path.name)[0])
            static = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}.get(target.path)
            if static:
                return self.send((self.static_root / static).read_bytes(), {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8"}[static])
            self.send({"error": "not found"}, status=404)
        except KeyError:
            self.send({"error": "not found"}, status=404)
        except (ValueError, TypeError):
            self.send({"error": "invalid request"}, status=400)
        except OSError:
            self.send({"error": "local source unavailable"}, status=503)

    def do_POST(self):
        self.send({"error": "read-only dashboard"}, status=405)

    do_PUT = do_POST
    do_DELETE = do_POST
    do_PATCH = do_POST


def create_server(store, port=8765):
    handler = type("BoundHandler", (Handler,), {"store": store})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print("Loading verified local corpus (read-only)...", flush=True)
    store = Observatory()
    if store.error:
        print(store.error, flush=True)
    server = create_server(store, args.port)
    print(f"Local: http://127.0.0.1:{server.server_port}/", flush=True)
    print("Loopback only; no GPU commands, uploads, source edits or external assets.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
