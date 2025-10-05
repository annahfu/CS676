from fastapi.testclient import TestClient
import server
client = TestClient(server.app)

def test_healthz():
    r = client.get('/healthz')
    assert r.status_code == 200
    assert 'status' in r.json()

def test_score_batch():
    payload = {'items':[
        {'id':'a','text':'Peer reviewed dataset','url':'https://example.edu'},
        {'id':'b','text':'Miracle cure!!!','url':'https://blogspot.example.com/post'}
    ]}
    r = client.post('/score_batch', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 'results' in body and len(body['results']) == 2
