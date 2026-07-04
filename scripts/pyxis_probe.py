#!/usr/bin/env python3
"""Standalone PYXIS / Blackmagic Camera Control REST API probe.

Run this against a camera IP to get a clear verdict on whether the REST
control API is actually answering -- separate from the app, so you can
re-check after toggling settings on the camera itself.

    python scripts/pyxis_probe.py 192.168.1.185
    python scripts/pyxis_probe.py 192.168.1.185 --port 80

It checks TCP reachability, identifies the device from its TLS
certificate (Blackmagic cameras present one on 443), and probes the
documented endpoints over both HTTP and HTTPS. A camera with the REST API
enabled answers GET /control/api/v1/system with HTTP 200 + JSON. The
"redirect-to-trailing-slash then 404" pattern on every endpoint means the
API tree is not mounted -- i.e. the REST API is disabled/secured on the
camera and must be enabled in its setup.
"""
from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import urllib.error
import urllib.request

ENDPOINTS = [
    "/system",
    "/transports/0/record",
    "/video/iso",
    "/video/whiteBalance",
    "/video/shutter",
    "/lens/zoom",
    "/media/active",
    "/system/format",
]


def tcp_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def tls_identity(host: str, port: int = 443) -> str | None:
    """Read the server cert's subject via openssl (present everywhere we
    run this) so a Blackmagic device shows up as e.g. 'CN=PYXIS-6K.local,
    O=Blackmagic Design'. Falls back to just confirming a cert is served."""
    import subprocess

    try:
        pem = ssl.get_server_certificate((host, port), timeout=3.0)
    except (OSError, TypeError):
        return None
    try:
        out = subprocess.run(
            ["openssl", "x509", "-noout", "-subject"],
            input=pem, capture_output=True, text=True, timeout=3.0, check=False,
        )
        subject = out.stdout.strip()
        return subject or "certificate present"
    except (OSError, subprocess.SubprocessError):
        return "certificate present"


def probe(scheme: str, host: str, port: int, path: str, timeout: float = 3.0) -> str:
    url = f"{scheme}://{host}:{port}/control/api/v1{path}"
    ctx = ssl._create_unverified_context() if scheme == "https" else None
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            body = resp.read(200)
            return f"HTTP {resp.status} {'JSON' if body.strip().startswith(b'{') else 'non-JSON'}"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError) as exc:
        return f"ERR {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("host")
    ap.add_argument("--port", type=int, default=80)
    args = ap.parse_args()

    print(f"== PYXIS REST API probe: {args.host} ==\n")

    print("TCP reachability:")
    for p in sorted({80, 443, args.port}):
        print(f"  :{p:<5} {'open' if tcp_open(args.host, p) else 'closed/filtered'}")

    ident = tls_identity(args.host)
    print(f"\nTLS device identity (:443): {ident or 'no certificate'}")

    verdict_ok = False
    for scheme, port in (("http", args.port), ("https", 443)):
        print(f"\n{scheme.upper()} :{port} endpoints:")
        for ep in ENDPOINTS:
            result = probe(scheme, args.host, port, ep)
            print(f"  {ep:<22} {result}")
            if result.startswith("HTTP 200"):
                verdict_ok = True

    print("\n== verdict ==")
    if verdict_ok:
        print("  REST API is ANSWERING (HTTP 200 on at least one endpoint). Good to go.")
        return 0
    print("  REST API is NOT answering. Every endpoint 307/404 or errors means the")
    print("  /control/api/v1 tree is not mounted -- enable the REST API in the")
    print("  camera's setup (and check whether it requires HTTPS + authentication).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
