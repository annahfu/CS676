
import server_ext as srv
from fastapi.testclient import TestClient

client = TestClient(srv.app)

def test_search_and_score_mock(monkeypatch):
    async def fake_ddg(query: str, k: int):
        return [
            {"title": "Doc A", "href": "https://example.edu/a", "body": "Methodology and dataset reported."},
            {"title": "Doc B", "href": "https://blog.example.com/b", "body": "Miracle cure!!!"}
        ][:k]

    async def fake_fetch(client, url: str):
        return "<html><body><p>Peer reviewed dataset with replication.</p></body></html>" if "example.edu" in url else "<html><body><p>Shocking miracle cure!!!</p></body></html>"

    def fake_extract(html: str):
        return "Peer reviewed dataset with replication." if "Peer" in html else "Shocking miracle cure!!!"

    monkeypatch.setattr(srv, "ddg_search", fake_ddg)
    monkeypatch.setattr(srv, "fetch_url", fake_fetch)
    monkeypatch.setattr(srv, "extract_text", fake_extract)

    r = client.post("/search_and_score", json={"query":"test", "k":2})
    assert r.status_code == 200
    body = r.json()
    assert "results" in body and len(body["results"]) == 2
    assert body["results"][0]["url"].startswith("https://example.edu")
    assert body["results"][0]["label"] in ("High","Medium","Low")
