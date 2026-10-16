const $ = (id) => document.getElementById(id);
const categories = {control:'Player control',scenes:'Cross-scene interaction',behavior:'Enemy behavior',arena:'RIGen-Arena',state:'Combat & state'};
const dialog = $('demo-dialog'), video = $('clip-video'), overview = $('overview-video');
let demos = [], current = null, currentIndex = -2, selectedIndex = 0, pendingSeek = null, opener = null;
const el = (tag, cls, text) => { const n = document.createElement(tag); if(cls) n.className = cls; if(text !== undefined) n.textContent=text; return n; };
function renderGallery(filter='all') {
  const grid = $('demo-grid'); grid.replaceChildren();
  const list = demos.filter(d=>filter==='all'||d.category===filter);
  for(const d of list) {
    const card=el('button','demo-card'); card.type='button'; card.setAttribute('aria-label',`${d.title}. Open video and chunk guidance`);
    const visual=el('div','card-visual'), img=el('img'); img.src=d.poster; img.alt=d.title; img.loading='lazy'; img.width=960;img.height=540;
    const line=el('div','card-topline');line.append(el('span','card-tag',categories[d.category]),el('span','card-duration',`${d.duration.toFixed(2)} s`));
    visual.append(img,line,el('span','card-play','▶'));
    const meta=el('div','card-meta');meta.append(el('h3',null,d.title));const label=el('span',null,d.chunks.length?`${d.chunks.length} chunks`:'Continuous clip');label.append(el('span','card-arrow','↗'));meta.append(label);
    card.append(visual,meta,el('p',null,d.description));card.addEventListener('click',()=>openDemo(d,card));grid.append(card);
  }
  $('gallery-status').textContent=`Showing ${list.length} demonstrations`;
}
function openDemo(d, trigger) {
  overview.pause();opener=trigger;current=d;currentIndex=-2;selectedIndex=0;pendingSeek=null;
  $('dialog-title').textContent=d.title;$('dialog-category').textContent=categories[d.category];
  $('clip-description').textContent=d.description;$('global-text').textContent=d.global;
  document.querySelector('.global-guidance').hidden=!d.global;
  document.querySelectorAll('dialog details').forEach(x=>x.open=false);
  $('loop-chunk').checked=false;$('loop-chunk').disabled=d.chunks.length===0;$('video-error').hidden=true;
  video.preload='auto';video.src=video.canPlayType('video/mp4; codecs="avc1.42E01E"')?d.video:d.video.replace('.mp4','.webm');video.poster=d.poster;video.load();
  const timeline=$('chunk-timeline');timeline.replaceChildren();timeline.hidden=d.chunks.length===0;timeline.classList.toggle('many',d.chunks.length>10);
  d.chunks.forEach((c,i)=>{const b=el('button','chunk-button',String(i+1).padStart(2,'0'));b.type='button';b.setAttribute('aria-label',`Chunk ${i+1}, ${c.start.toFixed(2)} to ${c.end.toFixed(2)} seconds`);b.title=`${c.start.toFixed(2)}–${c.end.toFixed(2)} s`;b.addEventListener('click',()=>seekChunk(i));timeline.append(b)});
  showChunk(-1);$('clip-time').textContent=`0.00 / ${d.duration.toFixed(2)} s`;
  dialog.showModal();document.body.classList.add('no-scroll');dialog.scrollTop=0;
}
function showChunk(i) {
  if(!current||i===currentIndex)return;currentIndex=i;
  document.querySelectorAll('.chunk-button').forEach((b,j)=>{b.classList.toggle('active',j===i);b.setAttribute('aria-pressed',String(j===i))});
  const c=current.chunks[i];
  $('guidance-title').textContent=c?`Chunk ${String(i+1).padStart(2,'0')}`:'Initial frame';
  $('guidance-range').textContent=c?`${c.start.toFixed(2)}–${c.end.toFixed(2)} s`:'0.00–0.05 s';
  $('guidance-text').replaceChildren(el('p',null,c?c.text:'The initial frame establishes the scene. Play the video or select a chunk below to inspect its guidance.'));
  $('full-text').textContent=c?c.full:current.global;
  $('chunk-count').textContent=c?`Chunk ${i+1} of ${current.chunks.length}`:(current.chunks.length?`${current.chunks.length} generation chunks`:'Continuous clip');
  $('prev-chunk').disabled=i<=0;$('next-chunk').disabled=i>=current.chunks.length-1;
}
function seekChunk(i, pause=true) {
  if(!current||i<0||i>=current.chunks.length)return;
  if(pause)video.pause();
  selectedIndex=i;pendingSeek=i;showChunk(i);performPendingSeek();
}
function performPendingSeek() {
  if(!current||pendingSeek===null||video.readyState<1||!Number.isFinite(video.duration))return;
  const target=Math.min(current.chunks[pendingSeek].start+0.001,Math.max(0,video.duration-0.001));
  // Do not require seekable[] to contain the target. Some static servers expose
  // the duration before the first byte-range response, but currentTime is still
  // the correct browser API for seeking once metadata is available.
  try {
    video.removeAttribute('poster');
    video.currentTime=target;
    pendingSeek=null;
  } catch (_) {}
}
function syncTime() {
  if(!current||pendingSeek!==null||video.seeking)return;
  let t=video.currentTime;
  if($('loop-chunk').checked&&current.chunks.length){const c=current.chunks[selectedIndex];if(t>=c.end-0.018||t<c.start-0.01){video.currentTime=c.start+0.001;t=video.currentTime;}}
  $('clip-time').textContent=`${t.toFixed(2)} / ${current.duration.toFixed(2)} s`;
  const i=current.chunks.findIndex(c=>t>=c.start-0.002&&t<c.end);
  showChunk(i<0&&t>=current.duration-0.025?current.chunks.length-1:i);
}
video.addEventListener('timeupdate',syncTime);video.addEventListener('seeked',syncTime);
['loadedmetadata','loadeddata','canplay','progress','canplaythrough'].forEach(event=>video.addEventListener(event,performPendingSeek));
// requestVideoFrameCallback keeps the guidance aligned even during rapid chunk transitions.
if('requestVideoFrameCallback' in video){const tick=()=>{syncTime();video.requestVideoFrameCallback(tick)};video.requestVideoFrameCallback(tick)}
video.addEventListener('error',()=>{if(current)$('video-error').hidden=false});
video.addEventListener('play',()=>overview.pause());
video.addEventListener('ended',()=>{if($('loop-chunk').checked&&current?.chunks.length){seekChunk(selectedIndex);video.play().catch(()=>{})}});
$('loop-chunk').addEventListener('change',()=>{if($('loop-chunk').checked&&current?.chunks.length){selectedIndex=Math.max(0,currentIndex);seekChunk(selectedIndex,false)}});
$('prev-chunk').addEventListener('click',()=>seekChunk(currentIndex-1));$('next-chunk').addEventListener('click',()=>seekChunk(currentIndex+1));
$('close-dialog').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close()}});
dialog.addEventListener('close',()=>{video.pause();current=null;pendingSeek=null;video.removeAttribute('src');video.load();document.body.classList.remove('no-scroll');opener?.focus()});
document.querySelectorAll('.filter').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.filter').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b))});renderGallery(b.dataset.filter)}));
$('film-play').addEventListener('click',()=>{overview.play().catch(()=>{$('film-play').hidden=false})});
overview.addEventListener('play',()=>{$('film-play').hidden=true;video.pause()});
overview.addEventListener('ended',()=>{$('film-play').hidden=false});
fetch('demos.json').then(r=>{if(!r.ok)throw Error('Failed to load demos');return r.json()}).then(data=>{demos=data;renderGallery()}).catch(()=>{$('demo-grid').replaceChildren(el('p','loading-text','Demonstrations could not load. Please refresh the page.'))});
