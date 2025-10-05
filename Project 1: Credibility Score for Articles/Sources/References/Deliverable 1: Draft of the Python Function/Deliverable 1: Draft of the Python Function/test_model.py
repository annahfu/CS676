import numpy as np
from credrefactor import CredConfig, train_bundle, rule_score

def test_rule_score_bounds():
    assert 0.0 <= rule_score('') <= 100.0

def test_train_bundle_small():
    pos = 'Peer reviewed dataset with replication and methodology details.'
    neg = 'Shocking miracle cure!!! You will not believe this hoax!!!'
    texts = [pos]*20 + [neg]*20
    y = np.array([85.0]*20 + [20.0]*20, dtype=float)
    cfg = CredConfig(max_features=3000, min_df=1, ngram_max=2)
    bundle = train_bundle(texts, y, cfg)
    assert bundle.thresholds[0] < bundle.thresholds[1]
