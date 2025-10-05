
/**
 * Credibility Widget (vanilla JS)
 * Renders a list of {rank, title, url, snippet, score, label, explanation}
 * into a container. Designed for chat UIs or side panels.
 *
 * Usage:
 *   <div id="cred-container"></div>
 *   <script src="credibility_widget.js"></script>
 *   <link rel="stylesheet" href="credibility_widget.css"/>
 *   <script>
 *     renderCredibilityList(document.getElementById('cred-container'), resultsArray);
 *   </script>
 */
function badgeColor(label) {
  switch ((label || '').toLowerCase()) {
    case 'high': return '#16a34a'; // green-600
    case 'medium': return '#ca8a04'; // amber-600
    case 'low': return '#dc2626'; // red-600
    default: return '#6b7280'; // gray-500
  }
}

function renderCredibilityList(container, results) {
  if (!container) return;
  container.innerHTML = '';
  const list = document.createElement('div');
  list.className = 'cred-list';

  (results || []).forEach((r) => {
    const card = document.createElement('div');
    card.className = 'cred-card';

    const header = document.createElement('div');
    header.className = 'cred-header';

    const rank = document.createElement('span');
    rank.className = 'cred-rank';
    rank.textContent = `#${r.rank || '-'}`;

    const title = document.createElement('a');
    title.className = 'cred-title';
    title.href = r.url || '#';
    title.target = '_blank';
    title.rel = 'noopener';
    title.textContent = r.title || r.url || 'Untitled';

    const badge = document.createElement('span');
    badge.className = 'cred-badge';
    const label = r.label || 'unavailable';
    const score = (typeof r.score === 'number') ? ` (${Math.round(r.score)})` : '';
    badge.textContent = `Credibility: ${label}${score}`;
    badge.style.background = badgeColor(label);

    header.appendChild(rank);
    header.appendChild(title);
    header.appendChild(badge);

    const snippet = document.createElement('div');
    snippet.className = 'cred-snippet';
    snippet.textContent = r.snippet || '';

    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = 'Why this score?';
    details.appendChild(summary);

    const expl = r.explanation || {};
    const dl = document.createElement('div');
    dl.className = 'cred-expl';

    function addRow(k, v) {
      const row = document.createElement('div');
      row.className = 'cred-row';
      const key = document.createElement('div');
      key.className = 'cred-key';
      key.textContent = k;
      const val = document.createElement('div');
      val.className = 'cred-val';
      val.textContent = (v === null || v === undefined) ? '—' : String(v);
      row.appendChild(key); row.appendChild(val);
      dl.appendChild(row);
    }

    addRow('ML score', expl.ml_score);
    addRow('Rules score', expl.rules_score);
    addRow('Domain nudge', expl.domain_nudge);
    addRow('Thresholds', Array.isArray(expl.thresholds) ? expl.thresholds.join(', ') : '—');

    const pos = expl.top_positive || [];
    const neg = expl.top_negative || [];
    if (pos.length || neg.length) {
      const fx = document.createElement('div');
      fx.className = 'cred-fx';
      const p = document.createElement('div');
      p.innerHTML = `<strong>Top positive</strong>: ${pos.map(x=>x[0]).slice(0,5).join(', ') || '—'}`;
      const n = document.createElement('div');
      n.innerHTML = `<strong>Top negative</strong>: ${neg.map(x=>x[0]).slice(0,5).join(', ') || '—'}`;
      fx.appendChild(p); fx.appendChild(n);
      dl.appendChild(fx);
    }

    details.appendChild(dl);

    card.appendChild(header);
    card.appendChild(snippet);
    card.appendChild(details);
    list.appendChild(card);
  });

  container.appendChild(list);
}

window.renderCredibilityList = renderCredibilityList;
