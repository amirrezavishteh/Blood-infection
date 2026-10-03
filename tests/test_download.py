"""Downloader against a local HTTP server that mimics the PhysioNet listing."""

import functools
import http.server
import threading

import pytest

from sepsis.data import download as dl
from sepsis.data.synthetic import make_fixture


class _Handler(http.server.SimpleHTTPRequestHandler):
    overrides: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in self.overrides:
            body = self.overrides[path]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.endswith("/"):
            d = self.translate_path(path)
            import os
            names = sorted(n for n in os.listdir(d) if n.endswith(".psv"))
            rows = [f'<a href="{n}">{n}</a>                 27-Mar-2019 18:57     {os.path.getsize(os.path.join(d, n))}'
                    for n in names]
            body = ("<html><body><pre>" + "\n".join(rows) + "</pre></body></html>").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    make_fixture(root, n_a=6, n_b=4)
    _Handler.overrides = {}
    handler = functools.partial(_Handler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/", root
    httpd.shutdown()


def test_download_resume_and_manifest(server, tmp_path):
    url, _ = server
    out = tmp_path / "out"
    m = dl.download_physionet2019(base_url=url, out_root=out, workers=3, limit=10)
    assert m["hospitals"]["A"]["present_files"] == 6
    assert m["hospitals"]["B"]["present_files"] == 4
    assert all(len(f["sha256"]) == 64 for f in m["hospitals"]["A"]["files"].values())
    assert (out / "manifest.json").exists()
    m2 = dl.download_physionet2019(base_url=url, out_root=out, workers=3, limit=10)
    assert m2["hospitals"]["A"]["transfer"] == {"downloaded": 0, "skipped": 6}


def test_download_fails_on_count_mismatch(server, tmp_path):
    url, _ = server
    with pytest.raises(dl.DownloadError, match="expected 20336"):
        dl.download_physionet2019(base_url=url, out_root=tmp_path / "o", workers=2)


def test_html_payload_rejected(server, tmp_path):
    url, root = server
    name = sorted((root / "training_setA").iterdir())[0].name
    size = (root / "training_setA" / name).stat().st_size
    fake = b"<!DOCTYPE html><html>" + b" " * (size - 21)
    _Handler.overrides = {f"/training_setA/{name}": fake}
    with pytest.raises(dl.DownloadError, match="failed"):
        dl.download_physionet2019(base_url=url, out_root=tmp_path / "o", workers=2, limit=10, retries=0)
    assert not (tmp_path / "o" / "training_setA" / name).exists()
