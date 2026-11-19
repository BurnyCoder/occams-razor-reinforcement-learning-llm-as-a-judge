/* Render only artifact data; candidate text always uses textContent, never HTML. */
let data = {runs: [], evaluations: [], examples: []};
const $ = id => document.getElementById(id);
const live = new URLSearchParams(location.search).has('live');
const format = (value, digits=3) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(digits);
function node(tag, text, parent) { const n=document.createElement(tag); n.textContent=text; if(parent) parent.append(n); return n; }
function chart(id, title, points, key, fixedMax) {
  const host=$(id); host.replaceChildren(); node('h3', title, host);
  const values=points.filter(p=>Number.isFinite(p[key]));
  if(!values.length){node('p','No recorded points yet.',host);return;}
  const ns='http://www.w3.org/2000/svg', svg=document.createElementNS(ns,'svg');
  svg.setAttribute('viewBox','0 0 320 170'); svg.setAttribute('role','img'); svg.setAttribute('aria-label',title);
  const low=Math.min(0,...values.map(p=>p[key])), high=fixedMax||Math.max(.01,...values.map(p=>p[key]));
  const xmax=Math.max(1,...values.map(p=>p.step));
  const x=p=>34+p.step/xmax*274, y=p=>137-(p[key]-low)/(high-low)*112;
  for(const value of [low,high]){const label=document.createElementNS(ns,'text');label.setAttribute('x','0');label.setAttribute('y',String(137-(value-low)/(high-low)*112));label.setAttribute('font-size','10');label.textContent=format(value,2);svg.append(label);}
  const line=document.createElementNS(ns,'polyline');line.setAttribute('points',values.map(p=>`${x(p)},${y(p)}`).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke','#286b52');line.setAttribute('stroke-width','2');svg.append(line);
  for(const p of values){const point=document.createElementNS(ns,'circle');point.setAttribute('cx',x(p));point.setAttribute('cy',y(p));point.setAttribute('r','2');point.setAttribute('fill','#286b52');svg.append(point);}
  const axis=document.createElementNS(ns,'text');axis.setAttribute('x','65');axis.setAttribute('y','162');axis.setAttribute('font-size','10');axis.textContent=`${key==='loss'?'Completed updates':'Updates before rollout'} (0–${xmax})`;svg.append(axis);host.append(svg);
}
function renderRun(){
  const run=data.runs.find(r=>r.name===$('run').value);$('cards').replaceChildren();
  const state=run?.status?.complete?`${run.status.pilot?'Completed pilot':'Completed'}${run.status.changed_parameter_tensors===0?'; no tensor changes':''}`:run?'In progress / interrupted':'Pending';
  for(const [label,value] of [['Training verifier pass rate',format(run?.training_accuracy)],['Rollouts / distinct tasks',run?`${run.rollouts} / ${run.unique_tasks??'unrecorded'}`:'—'],['Run state',state]]){const c=node('div','',$('cards'));c.className='card';node('span',label,c);node('strong',String(value),c);}
  chart('accuracy-chart','Rollout accuracy',run?.curve||[],'accuracy',1);
  chart('reward-chart','Mean group reward',run?.curve||[],'reward');
  chart('loss-chart','Optimization loss',run?.metrics||[],'loss');
}
function renderScores(){
  const body=$('scores');body.replaceChildren();
  const rows=data.evaluations.filter(r=>r.domain===$('domain').value);
  for(const r of rows){const tr=node('tr','',body);const state=(r.complete?'Complete':'Partial / unverified')+(r.exploratory?'; exploratory':'');for(const value of [r.run,r.split,r.n,format(r.correct),format(r.adequate),format(r.quality),format(r.tokens,1),state])node('td',String(value),tr);tr.title=(r.evidence_issues||[]).join('; ');}
  if(!rows.length){const message=data.evaluations.length?'No results for this selection. Select an available domain; an overall score requires every domain.':'No evaluation results have been published.';const td=node('td',message,node('tr','',body));td.colSpan=8;}
  $('analysis').textContent=`Primary paired analysis: ${data.analysis?.status||'pending'}. Accuracy preservation requires the registered confidence interval criteria, including each domain.`;
}
function renderExample(){const r=data.examples?.[Number($('example').value)];$('trace').textContent=r?.trace||'No evaluated examples yet.';$('example-info').textContent=r?`${r.arm} · ${r.domain} · ${r.task_id} · correct ${r.correct} · adequate ${r.adequate} · quality ${format(r.quality)}`:'';}
function renderAudits(){
  const profile=data.readiness?.profile;
  $('profile').textContent=profile?`Recorded synthetic profile: ${profile.passed?'passed':'failed'}. Minimum sampled free VRAM: backward ${profile.backward_minimum_free_mib} MiB; generation ${profile.generation_minimum_free_mib} MiB. Role-switch trainable tensors and CUDA RNG equal: ${profile.switch_weights_unchanged&&profile.switch_rng_unchanged?'yes':'no'}. Optimizer state-entry count preserved after reload: ${profile.optimizer_roundtrip?'yes':'no'}. Interrupted-training resumption remains unverified.`:'No completed GPU profile yet.';
  const body=$('audits');body.replaceChildren();
  const audits=data.readiness?.audits||[];
  for(const audit of audits){const label=audit.model+(audit.audit_file?` (${audit.audit_file.split('/').pop().replace('judge-audit-','').replace('.json','')})`:'');const tr=node('tr','',body);for(const value of [label,`${audit.wins}/${audit.pair_count}`,`${audit.rejected}/${audit.omission_count}`,audit.invalid,audit.passed?'Yes':'No'])node('td',String(value),tr);}
  if(!audits.length){const td=node('td','No completed judge audits yet.',node('tr','',body));td.colSpan=5;}
}
async function refresh(){
  try{const response=await fetch(live?'live.json':'data.json',{cache:'no-store'});if(!response.ok)throw Error(`HTTP ${response.status}`);data=await response.json();
    $('status').textContent=`${live?'Live local dashboard':'Published snapshot'} · ${data.generated_at||'No experiments yet'}`;
    const previous=$('run').value;$('run').replaceChildren();for(const r of data.runs){const option=node('option',r.name,$('run'));option.value=r.name;}if(!data.runs.length)node('option','No runs yet',$('run'));if(data.runs.some(r=>r.name===previous))$('run').value=previous;
    $('example').replaceChildren();(data.examples||[]).forEach((r,i)=>{const option=node('option',`${r.arm} / ${r.task_id}`,$('example'));option.value=String(i);});
    if(!data.examples?.length)node('option','No evaluated examples yet',$('example'));
    renderAudits();renderRun();renderScores();renderExample();
  }catch(error){$('status').textContent=`Artifacts unavailable: ${error.message}`;}
}
$('run').addEventListener('change',renderRun);$('domain').addEventListener('change',renderScores);$('example').addEventListener('change',renderExample);
refresh();if(live)setInterval(refresh,5000);
