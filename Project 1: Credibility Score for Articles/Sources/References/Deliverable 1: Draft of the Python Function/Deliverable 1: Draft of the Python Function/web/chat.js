function el(tag, cls, text){
  const e = document.createElement(tag);
  if(cls) e.className = cls;
  if(text!==undefined) e.textContent = text;
  return e;
}
const chat = document.getElementById('chat');
const msgInput = document.getElementById('msg');
const sendBtn = document.getElementById('send');

function addUserBubble(text){
  const b = el('div','msg user');
  b.appendChild(el('div','small','You'));
  b.appendChild(el('div',null,text));
  chat.appendChild(b);
  chat.scrollTop = chat.scrollHeight;
}
function addTyping(){
  const t = el('div','typing','Thinking…');
  chat.appendChild(t);
  chat.scrollTop = chat.scrollHeight;
  return t;
}
function badge(label, score){
  const s = typeof score==='number' ? ` (${Math.round(score)})` : '';
  const span = el('span','badge na',`Credibility: ${label||'N/A'}${s}`);
  if(!label){ span.className='badge na'; return span; }
  const L = label.toLowerCase();
  span.className = 'badge ' + (L==='high'?'high':(L==='medium'?'medium':(L==='low'?'low':'na')));
  return span;
}
function renderExplanation(ex){
  const box = el('div','kv');
  function row(k,v){ const a=el('div','small',k), b=el('div','small',v==null?'—':String(v)); box.append(a,b); }
  row('ML score', ex?.ml_score);
  row('Rules score', ex?.rules_score);
  row('Domain nudge', ex?.domain_nudge);
  row('Thresholds', Array.isArray(ex?.thresholds)?ex.thresholds.join(', '):'—');
  if((ex?.top_positive?.length||0)+(ex?.top_negative?.length||0)>0){
    const fx = el('div','fx');
    fx.innerHTML = `<div class="small"><b>Top positive</b>: ${(ex.top_positive||[]).map(x=>x[0]).slice(0,5).join(', ')||'—'}</div>
                    <div class="small"><b>Top negative</b>: ${(ex.top_negative||[]).map(x=>x[0]).slice(0,5).join(', ')||'—'}</div>`;
    box.appendChild(fx);
  }
  return box;
}
function addAssistantBubble(contentNodes){
  const b = el('div','msg assistant');
  b.appendChild(el('div','small','Assistant'));
  contentNodes.forEach(n => b.appendChild(n));
  chat.appendChild(b);
  chat.scrollTop = chat.scrollHeight;
}
async function callAPI(mode, text){
  if(mode==='search'){
    const r = await fetch('/search_and_score', {
      method: 'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({query: text, k: 5})
    });
    if(!r.ok) throw new Error('API error: '+ r.status);
    return await r.json();
  }else{
    const r = await fetch('/score', {
      method: 'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({id:'chat', text, url: null})
    });
    if(!r.ok) throw new Error('API error: '+ r.status);
    return await r.json();
  }
}
function renderSearchResults(arr){
  const wrap = el('div');
  if(!arr?.length){ wrap.append(el('div','small','No results.')); return [wrap]; }
  arr.forEach(r => {
    const card = el('div','card');
    const header = el('div','row');
    const title = el('a',null,r.title||r.url||'Untitled');
    title.href = r.url||'#'; title.target='_blank'; title.rel='noopener';
    header.appendChild(title);
    header.appendChild(badge(r.label, r.score));
    card.appendChild(header);
    if(r.snippet) card.appendChild(el('div','small',r.snippet));
    if(r.explanation) card.appendChild(renderExplanation(r.explanation));
    wrap.appendChild(card);
  });
  return [wrap];
}
function renderScoreResult(obj){
  const nodes = [];
  nodes.push(badge(obj?.results?.[0]?.label, obj?.results?.[0]?.score));
  if(obj?.results?.[0]?.explanation){
    nodes.push(renderExplanation(obj.results[0].explanation));
  }
  return nodes;
}
sendBtn.addEventListener('click', async () => {
  const text = msgInput.value.trim();
  if(!text) return;
  const mode = (document.querySelector('input[name="mode"]:checked')?.value)||'search';
  addUserBubble(text);
  msgInput.value=''; msgInput.focus();
  const typing = addTyping();
  try{
    const data = await callAPI(mode, text);
    typing.remove();
    if(mode==='search'){
      addAssistantBubble(renderSearchResults(data.results));
    }else{
      addAssistantBubble(renderScoreResult(data));
    }
  }catch(e){
    typing.remove();
    addAssistantBubble([el('div','small','Sorry—something went wrong: ' + e.message)]);
  }
});
msgInput.addEventListener('keydown', e => {
  if(e.key==='Enter' && !e.shiftKey){
    e.preventDefault();
    sendBtn.click();
  }
});
