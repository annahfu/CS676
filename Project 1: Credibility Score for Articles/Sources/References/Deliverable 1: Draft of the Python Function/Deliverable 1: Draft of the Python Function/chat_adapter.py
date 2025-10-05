import os, requests

CRED_API = os.getenv('CRED_API_URL', 'http://localhost:7860')

def score_for_chat(user_text: str, url: str | None = None) -> dict:
    payload = {'id':'chat','text':user_text,'url':url}
    r = requests.post(f'{CRED_API}/score', json=payload, timeout=6)
    r.raise_for_status()
    out = r.json()['results'][0]
    return {
        'score': round(out['score'] or 0),
        'label': out['label'],
        'explanation': out.get('explanation', {}),
        'fallback': ('fallback' in (out.get('errors') or []))
    }
