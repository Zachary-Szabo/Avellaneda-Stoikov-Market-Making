"""Download historical CF Benchmarks BRTI values through the Kalshi passthrough.

Signing follows Kalshi's authenticated-request guide: the message is
timestamp + METHOD + path, the path has no query string, RSA keys use
RSA-PSS with SHA-256, and Ed25519 keys sign the message directly.
"""
import argparse
import base64
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dotenv import load_dotenv

BASE_URL = "https://external-api.kalshi.com"
SIGN_PATH = "/trade-api/v2/cfbenchmarks/history/values"
OUT_DIR = Path("data/brti")
# Basic-tier reads refill at 200 tokens/s and this endpoint costs 50, so
# about four requests per second is the sustained budget. Stay under that.
PAUSE_S = 0.3
MAX_ATTEMPTS = 6


def load_credentials() -> tuple[str, object]:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    key_id = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
    missing = []
    if not key_id:
        missing.append("KALSHI_API_KEY_ID")
    if not key_path:
        missing.append("KALSHI_PRIVATE_KEY_PATH")
    if missing:
        sys.exit("Missing " + " and ".join(missing) + " in .env")

    path = Path(key_path)
    if not path.is_file():
        sys.exit("KALSHI_PRIVATE_KEY_PATH does not point to a file")
    with path.open("rb") as handle:
        private_key = serialization.load_pem_private_key(
            handle.read(), password=None, backend=default_backend()
        )
    return key_id, private_key


def sign_request(private_key, timestamp: str, method: str, path: str) -> str:
    """Kalshi request signature. `path` is signed without any query string."""
    path_without_query = path.split("?")[0]
    message = f"{timestamp}{method}{path_without_query}".encode("utf-8")
    if isinstance(private_key, Ed25519PrivateKey):
        signature = private_key.sign(message)
    else:
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
    return base64.b64encode(signature).decode("utf-8")


def auth_headers(private_key, key_id: str) -> dict[str, str]:
    timestamp = str(int(datetime.now().timestamp() * 1000))
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sign_request(private_key, timestamp, "GET", SIGN_PATH),
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }


def hour_stamp(hour: datetime) -> str:
    """ISO time truncated to the hour, which the history endpoint requires."""
    return hour.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00.000Z")


def fetch_hour(private_key, key_id: str, hour: datetime) -> dict:
    params = {"id": "BRTI", "timespan": "HOUR", "timestamp": hour_stamp(hour)}
    url = BASE_URL + SIGN_PATH
    delay = 1.0
    last_status = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = requests.get(
            url,
            params=params,
            headers=auth_headers(private_key, key_id),
            timeout=120,
        )
        if response.status_code == 429 or response.status_code >= 500:
            last_status = response.status_code
            if attempt == MAX_ATTEMPTS:
                sys.exit(f"HTTP {last_status} for {params['timestamp']} after {MAX_ATTEMPTS} attempts")
            retry_after = response.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.isdigit() else delay
            print(f"HTTP {response.status_code} for {params['timestamp']}, retrying in {wait:.0f}s")
            time.sleep(wait)
            delay *= 2
            continue
        if not response.ok:
            detail = response.text[:300].replace("\n", " ")
            sys.exit(f"HTTP {response.status_code} for {params['timestamp']}: {detail}")
        try:
            body = response.json()
        except ValueError:
            sys.exit(f"Non-JSON response for {params['timestamp']}")
        if not isinstance(body, dict):
            sys.exit(f"Expected a JSON object for {params['timestamp']}")
        return body
    sys.exit(f"HTTP {last_status} for {params['timestamp']}")


def envelope(body: dict) -> dict:
    data = body.get("data")
    return data if isinstance(data, dict) else body


def extract_points(body: dict) -> list[dict]:
    inner = envelope(body)
    error = inner.get("error")
    if error:
        sys.exit(f"CF Benchmarks error: {error}")
    payload = inner.get("payload")
    if not isinstance(payload, list):
        sys.exit("Response payload is not a list. Run with --probe and inspect data/brti/probe.json.")
    points = []
    for item in payload:
        if not isinstance(item, dict):
            sys.exit("Payload items are not objects. Run with --probe and inspect data/brti/probe.json.")
        points.append(item)
    return points


def as_utc(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # int64 values from this API are unix milliseconds; smaller numbers are seconds.
    seconds = value / 1000.0 if abs(value) > 1e11 else float(value)
    try:
        return datetime.fromtimestamp(seconds, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def format_time(value) -> str:
    parsed = as_utc(value)
    if parsed is None:
        return str(value)
    return f"{value} ({parsed.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z)"


def format_spacing(times: list) -> str:
    diffs = []
    for earlier, later in zip(times, times[1:]):
        if isinstance(earlier, bool) or isinstance(later, bool):
            continue
        if isinstance(earlier, (int, float)) and isinstance(later, (int, float)):
            diffs.append(later - earlier)
    if not diffs:
        return "timestamps are not numeric, so spacing was not computed"
    diffs.sort()
    unit = "ms" if abs(times[0]) > 1e11 else "s"
    median = diffs[len(diffs) // 2]
    if diffs[0] == diffs[-1]:
        return f"{diffs[0]:g} {unit}, constant across {len(diffs)} gaps"
    return (
        f"min {diffs[0]:g} {unit}, median {median:g} {unit}, "
        f"max {diffs[-1]:g} {unit} across {len(diffs)} gaps"
    )


def describe(body: dict) -> None:
    print("top-level keys:", ", ".join(body))
    inner = envelope(body)
    if inner is not body:
        print("data keys:", ", ".join(inner))
    payload = inner.get("payload")
    if not isinstance(payload, list):
        print("payload type:", type(payload).__name__)
        return
    print("points:", len(payload))
    if not payload:
        return
    sample = payload[0]
    if not isinstance(sample, dict):
        print("point type:", type(sample).__name__)
        return
    print("point keys:", ", ".join(sample))
    times = [row["time"] for row in payload if isinstance(row, dict) and "time" in row]
    if not times:
        print("no time field on the points")
        return
    print("first time:", format_time(times[0]))
    print("last time:", format_time(times[-1]))
    print("spacing:", format_spacing(times))


def write_day(path: Path, rows: list[dict]) -> int:
    fieldnames: list[str] = []
    for preferred in ("time", "value"):
        if any(preferred in row for row in rows):
            fieldnames.append(preferred)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if "time" in fieldnames:
        def sort_key(row: dict):
            stamp = row.get("time")
            if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
                return (0, stamp)
            if stamp is None:
                return (2, "")
            return (1, str(stamp))

        rows = sorted(rows, key=sort_key)
        kept = []
        seen = set()
        for row in rows:
            stamp = row.get("time")
            if stamp in seen:
                continue
            seen.add(stamp)
            kept.append(row)
        rows = kept
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_day(text: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        sys.exit(f"Expected YYYY-MM-DD, got {text!r}")


def default_probe_hour() -> datetime:
    """An hour whose window ended at least an hour ago, so history can be published."""
    closed = datetime.now(timezone.utc) - timedelta(hours=2)
    return closed.replace(minute=0, second=0, microsecond=0)


def download_days(private_key, key_id: str, start: datetime, end: datetime) -> None:
    if end < start:
        sys.exit("--end is before --start")
    day = start
    while day <= end:
        dest = OUT_DIR / f"{day:%Y-%m-%d}.csv"
        if dest.exists():
            print(f"skip {day:%Y-%m-%d} (already saved)")
            day += timedelta(days=1)
            continue
        rows: list[dict] = []
        for hour in range(24):
            stamp = day + timedelta(hours=hour)
            body = fetch_hour(private_key, key_id, stamp)
            points = extract_points(body)
            rows.extend(points)
            print(f"{hour_stamp(stamp)}  {len(points)} points")
            time.sleep(PAUSE_S)
        if not rows:
            print(f"{day:%Y-%m-%d} returned no points; not writing a CSV")
        else:
            written = write_day(dest, rows)
            print(f"wrote {dest} ({written} rows)")
        day += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical BRTI values via Kalshi.")
    parser.add_argument("--start", help="First UTC day, YYYY-MM-DD, inclusive")
    parser.add_argument("--end", help="Last UTC day, YYYY-MM-DD, inclusive")
    parser.add_argument("--probe", action="store_true", help="Fetch one hour, print its shape, and save the raw JSON")
    args = parser.parse_args()

    key_id, private_key = load_credentials()
    if args.probe:
        hour = parse_day(args.start) if args.start else default_probe_hour()
        body = fetch_hour(private_key, key_id, hour)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = OUT_DIR / "probe.json"
        dest.write_text(json.dumps(body, indent=2), encoding="utf-8")
        print(f"probed {hour_stamp(hour)}")
        describe(body)
        print(f"saved {dest}")
        return

    if not args.start or not args.end:
        parser.error("--start and --end are required unless --probe is set")
    download_days(private_key, key_id, parse_day(args.start), parse_day(args.end))


if __name__ == "__main__":
    main()
