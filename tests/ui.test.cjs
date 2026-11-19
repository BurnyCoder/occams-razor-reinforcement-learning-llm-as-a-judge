/* Exercise the shipped rendering code with synthetic records, never model outputs.
 * Node's test runner: https://nodejs.org/api/test.html
 * A small DOM double tests labels/events; browser inspection checks actual layout.
 */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this.textContent='';this.value='';this.events={};this.attributes={};}
  append(child){this.children.push(child);if(this.tag==='select'&&this.children.length===1)this.value=child.value;}
  replaceChildren(){this.children=[];this.textContent='';}
  setAttribute(k,v){this.attributes[k]=String(v);}
  addEventListener(k,fn){this.events[k]=fn;}
  get text(){return this.textContent+this.children.map(c=>c.text).join(' ');}
}

async function page(payload, live=false){
  const elements=new Map();
  const get=id=>{if(!elements.has(id))elements.set(id,new Element(['domain','run','example'].includes(id)?'select':'div'));return elements.get(id);};
  get('domain').value='all';
  const urls=[];
  const context=vm.createContext({document:{getElementById:get,createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)},
    location:{search:live?'?live=1':''},URLSearchParams,setInterval:()=>{},
    fetch:async url=>{urls.push(url);return {ok:true,json:async()=>payload};}});
  vm.runInContext(readFileSync('site/app.js','utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  return {get,context,urls};
}

test('empty and partial domains remain explicit; candidates are plain text',async()=>{
  const payload={runs:[],evaluations:[],examples:[]};
  const p=await page(payload);
  assert.match(p.get('scores').text,/No evaluation/);
  assert.match(p.get('cards').text,/Pending/);
  payload.evaluations.push({domain:'math',run:'fixture',split:'test',n:1,correct:1,adequate:1,quality:1,tokens:2,complete:false,exploratory:true,evidence_issues:['missing tasks']});
  payload.examples.push({arm:'fixture',domain:'math',task_id:'a',trace:'<img src=x onerror=alert(1)>',correct:1,adequate:1,quality:1});
  vm.runInContext('renderScores()',p.context);
  assert.match(p.get('scores').text,/overall score requires every domain/);
  p.get('domain').value='math';p.get('domain').events.change();
  assert.match(p.get('scores').text,/Partial \/ unverified; exploratory/);
  p.get('example').value='0';p.get('example').events.change();
  assert.equal(p.get('trace').textContent,payload.examples[0].trace);
  assert.equal(p.get('trace').children.length,0);
});

test('live completed pilot and chart indices do not imply learning',async()=>{
  const run={name:'pilot',status:{complete:true,pilot:true,changed_parameter_tensors:0},rollouts:4,unique_tasks:1,training_accuracy:0,
    curve:[{step:0,accuracy:0,reward:0}],metrics:[{step:1,loss:0}]};
  const p=await page({runs:[run],evaluations:[],examples:[]},true);
  p.get('run').value='pilot';p.get('run').events.change();
  assert.deepEqual(p.urls,['live.json']);
  assert.match(p.get('cards').text,/Completed pilot; no tensor changes/);
  assert.match(p.get('cards').text,/4 \/ 1/);
  assert.match(p.get('accuracy-chart').text,/Updates before rollout/);
  assert.match(p.get('loss-chart').text,/Completed updates/);
  assert.equal(p.get('accuracy-chart').children[1].children.filter(c=>c.tag==='circle').length,1);
});
