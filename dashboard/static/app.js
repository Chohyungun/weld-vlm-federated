"use strict";
const $ = (id) => document.getElementById(id);
const num = (x) => x == null ? "—" : Number(x).toLocaleString("ko-KR");
const esc = (x) => String(x ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const time = (x) => { if (!x) return "—"; const d = new Date(x); return Number.isNaN(+d) ? "—" : new Intl.DateTimeFormat("ko-KR", {timeZone:"Asia/Seoul",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}).format(d); };
const labels = {pending:"미회수 / 미판정", flagged:"검토 필요", all_true:"3항목 true", parse_error:"파싱 오류"};
let selected = null, offset = 0, total = 0, requestNo = 0, detailNo = 0, summaryData = null;
async function api(path) { const response = await fetch(path, {cache:"no-store"}); if (!response.ok) throw new Error(`자료 요청 실패 (${response.status})`); return response.json(); }
function error(message) { $("error").textContent = message; $("error").hidden = !message; }
function bar(name, value, maximum, color = "") { const width = Math.min(100, Math.max(0, maximum ? value / maximum * 100 : 0)); return `<div class="bar-item"><div class="bar-label"><span>${esc(name)}</span><strong>${num(value)}</strong></div><div class="bar-track"><span style="width:${width}%;${color ? `background:${color}` : ""}"></span></div></div>`; }
function renderSummary(data) {
  if (data.error) throw new Error(data.error);
  summaryData = data;
  const live = data.live, fresh = live.available && !live.stale;
  const gen = live.available && live.generated_rows != null ? live.generated_rows : data.generated_rows;
  const judged = live.available && live.judged_rows != null ? live.judged_rows : data.judged_rows;
  const gs = live.available && live.generated_shards != null ? live.generated_shards : data.generated_shards;
  const js = live.available && live.judged_shards != null ? live.judged_shards : data.judged_shards;
  const cost = live.available && live.cost_estimate_won != null ? live.cost_estimate_won : data.cost_estimate_won;
  $("run-id").textContent = data.run_id;
  $("observed").textContent = `관측 ${time(live.available ? live.observed_at_utc : data.snapshot_at)} KST`;
  $("connection").className = `connection${fresh ? " live" : ""}`;
  $("connection").textContent = fresh ? "서버 관측 연결" : live.available ? "마지막 관측 · STALE" : "로컬 회수 스냅샷";
  $("freshness").classList.toggle("stale", !fresh);
  $("freshness").innerHTML = `<span>ⓘ</span><div>${fresh ? `서버 상태는 30초 주기 관측값입니다. 마지막 관측 ${time(live.observed_at_utc)} KST. 샘플 탐색은 10월 9일에 해시 검증·회수한 결과를 보여줍니다.` : live.available ? `서버 관측이 ${num(live.age_seconds)}초 전입니다. 아래 운영 수치는 마지막 관측값이며, 현재 실행 상태를 보장하지 않습니다. 샘플은 검증된 회수 자료입니다.` : `라이브 서버 관측은 아직 연결되지 않았습니다. ${time(data.snapshot_at)} KST에 저장된 회수 결과를 표시하며 현재 GPU 상태로 해석하지 않습니다.`}${live.poll_error ? " 서버 연결 점검 실패 기록이 있습니다." : ""}</div>`;
  const observedSuffix = live.available && !fresh ? " · 마지막 관측" : "";
  $("stats").innerHTML = [
    ["전체 데이터 작업", data.tasks, "행", "train + val · 영상 설명 전용", ""],
    ["생성 완료" + observedSuffix, gen, "행", `${num(gs)} / ${num(data.total_shards)} 샤드 · Qwen 2.5`, "primary"],
    ["자동 판정" + observedSuffix, judged, "행", live.available && live.validated_judged_shards != null ? `${num(js)} 관측 / ${num(live.validated_judged_shards)} 검증 샤드 · 전체 ${num(data.total_shards)}` : `${num(js)} / ${num(data.total_shards)} 샤드 · Phi 4 mini`, ""],
    ["판정 잔여" + observedSuffix, Math.max(0, data.tasks - judged), "행", "완료 샤드는 재생성하지 않음", ""],
  ].map(([title,value,unit,note,cls]) => `<article class="stat ${cls}"><i class="stat-dot"></i><p>${esc(title)}</p><strong>${num(value)}<span>${esc(unit)}</span></strong><small>${esc(note)}</small></article>`).join("");
  const stageLabels = [["데이터 준비", "명부 · 해시 고정", "done"], ["설명 생성", "전량 저장 확인", "done"], ["자동 판정", fresh && live.supervisor_running ? "서버 작업 중" : "일부 완료 · 관측 기준", fresh && live.supervisor_running ? "current" : ""], ["사람 검수", "별도 검토 필요", ""], ["정식 릴리스", "품질 보류", ""], ["모델 학습", "미실행", ""]];
  $("pipeline").innerHTML = stageLabels.map(([name,note,cls],i) => `<div class="step ${cls}"><span class="step-num">${cls === "done" ? "✓" : String(i+1).padStart(2,"0")}</span><strong>${name}</strong><small>${note}</small></div>`).join("");
  const train = data.splits.train || 0, val = data.splits.val || 0, pct = train / data.tasks * 100;
  $("split-donut").innerHTML = `<svg viewBox="0 0 140 140" role="img" aria-label="train ${num(train)}행, val ${num(val)}행"><circle cx="70" cy="70" r="53" fill="none" stroke="#d5b974" stroke-width="13"/><circle cx="70" cy="70" r="53" fill="none" stroke="#348f79" stroke-width="13" pathLength="100" stroke-dasharray="${pct} 100" transform="rotate(-90 70 70)"/><text x="70" y="66" text-anchor="middle" fill="#2b5147" font-size="20" font-weight="600">${num(data.tasks)}</text><text x="70" y="86" text-anchor="middle" fill="#96a38d" font-size="8">DATA ROWS</text></svg>`;
  $("split-legend").innerHTML = `<div><i></i><span>Train <small>학습 후보</small></span><strong>${num(train)}</strong></div><div><i class="val"></i><span>Validation</span><strong>${num(val)}</strong></div><div><i class="qa"></i><span>Source QA</span><strong>${num((data.source_qa.train || 0) + (data.source_qa.val || 0))}</strong></div>`;
  $("clients").innerHTML = Object.entries(data.clients).map(([k,v]) => `<div>${esc(k)}<strong>${num(v)}</strong><div class="mini-meter"><span style="width:${v / data.tasks * 100}%"></span></div></div>`).join("");
  $("cost").textContent = `${num(cost)}원`;
  $("cost-bar").style.width = `${Math.min(100, Math.max(0, cost / data.cap_won * 100))}%`;
  $("cost-basis").textContent = live.available ? "누적 비용 + 현재 세션 보수 추정" : "이전 회차 종료 상한 추정";
  $("cost-time").textContent = time(live.available ? live.observed_at_utc : data.cost_as_of) + " KST";
  $("gpu").textContent = fresh && live.gpu && live.gpu.utilization_pct != null ? `${num(live.gpu.utilization_pct)}% · ${num(live.gpu.memory_used_mb)} MiB` : "현재 상태 확인 불가";
  const sorted = Object.entries(data.classes).sort((a,b) => b[1]-a[1]);
  $("class-bars").innerHTML = [bar("무결함 (결함 클래스 아님)", data.normal_rows, data.tasks), ...sorted.map(([k,v]) => bar(k,v,data.tasks))].join("");
  $("judge-bars").innerHTML = ["flagged","all_true","parse_error"].map(k => bar(labels[k],data.judge_counts[k] || 0,data.judged_rows,k === "flagged" ? "#cbb279" : "")).join("") + `<p class="footnote">회수된 ${num(data.judged_rows)}행 기준 · 최신 서버 누적과 다를 수 있습니다.</p>`;
  drawChart(data.stage_metrics);
  $("stage-metrics").innerHTML = [["generate","생성"],["judge","판정"]].map(([k,name]) => `<div><span>${name} · 샤드 중앙값</span><strong>${num(data.stage_metrics[k].median_shard_seconds)}<small> 초</small></strong><small>출력 ${num(data.stage_metrics[k].output_tokens)} tokens · peak ${num(data.stage_metrics[k].peak_reserved_gb)} GiB</small></div>`).join("");
  $("training-note").textContent = data.training.note;
  $("evidence").innerHTML = data.evidence.map(x => `<article class="evidence-item"><span>✓ ${esc(x.status)}</span><strong>${esc(x.name)}</strong><p>${esc(x.scope)}</p></article>`).join("");
  $("identity").textContent = `RUN IDENTITY · ${data.identity} / CODE · ${data.code} / 원본 SHA-256 검증 완료`;
}
function drawChart(metrics) {
  const width=650,height=190,pad=35, all=[...metrics.generate.series,...metrics.judge.series];
  const maxX=Math.max(1,...all.map(p=>p.x)), maxY=Math.max(1,...all.map(p=>p.y || 0));
  let markup=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="생성과 판정의 샤드별 처리 시간"><title>저장 샤드별 처리 시간 (초)</title>`;
  for(let i=0;i<=3;i++){const y=15+(height-pad-15)*i/3;markup+=`<line x1="${pad}" y1="${y}" x2="${width-10}" y2="${y}" stroke="#ebefe5"/><text x="${pad-8}" y="${y+3}" text-anchor="end" fill="#a3ad96" font-size="8">${Math.round(maxY*(1-i/3))}s</text>`;}
  for(const [key,color] of [["generate","#559c7b"],["judge","#c4a25f"]]){const points=metrics[key].series.map(p=>`${pad+(width-pad-10)*p.x/maxX},${15+(height-pad-15)*(1-(p.y||0)/maxY)}`).join(" ");markup+=`<polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.5"/>`;}
  markup+=`<text x="${pad}" y="${height-7}" fill="#a3ad96" font-size="8">0</text><text x="${width-10}" y="${height-7}" text-anchor="end" fill="#a3ad96" font-size="8">${maxX} shard index</text></svg>`;
  $("throughput-chart").innerHTML=markup;
}
async function loadSamples(reset=true) {
  if(reset) offset=0;
  const request=++requestNo, params=new URLSearchParams({offset:String(offset),limit:"12"});
  for(const key of ["split","client","defect","judge"])if($(key).value)params.set(key,$(key).value);
  if($("search").value.trim())params.set("q",$("search").value.trim());
  try {
    const data=await api(`/api/samples?${params}`); if(request!==requestNo)return;
    total=data.total;
    $("sample-count").textContent=`${num(total)}개 작업`;
    $("page-info").textContent=total?`${Math.floor(offset/12)+1} / ${Math.ceil(total/12)}`:"0 / 0";
    $("prev").disabled=offset===0;$("next").disabled=offset+12>=total;
    $("sample-list").innerHTML=data.items.length?data.items.map(item=>`<button class="sample-item${item.id===selected?" active":""}" data-id="${esc(item.id)}"><strong>${esc(item.id.split(":").slice(1).join(" / "))}</strong><p>${esc(item.text || "생성 결과 없음")}</p><div class="sample-meta"><span>${esc(item.split)}</span><span>·</span><span>${esc(item.client)}</span><span>·</span><span>${Object.keys(item.defects).length ? esc(Object.keys(item.defects).join(" · ")) : "무결함"}</span><span class="status ${item.judge.status}">${labels[item.judge.status]}</span></div></button>`).join(""):`<div class="empty-state">조건에 맞는 데이터가 없습니다.<br>검색어나 필터를 바꿔보세요.</div>`;
    $("sample-list").querySelectorAll("button").forEach(button=>button.addEventListener("click",()=>loadDetail(button.dataset.id)));
    if(data.items.length && (!selected || !data.items.some(x=>x.id===selected)))await loadDetail(data.items[0].id);
    if(!data.items.length){selected=null;$("sample-detail").innerHTML=`<div class="empty-state">표시할 샘플이 없습니다.</div>`;}
  } catch(e) { error(e.message); }
}
async function loadDetail(id) {
  const request=++detailNo;selected=id;
  $("sample-list").querySelectorAll("button").forEach(b=>b.classList.toggle("active",b.dataset.id===id));
  try {
    const item=await api(`/api/sample?id=${encodeURIComponent(id)}`);if(request!==detailNo)return;
    const j=item.judge;
    $("sample-detail").innerHTML=`<div class="detail-header"><div><h3>${esc(item.id.split(":").slice(1).join(" / "))}</h3><div class="detail-tags"><span>${esc(item.split)}</span><span>${esc(item.client)}</span><span>${Object.entries(item.defects).map(([k,v])=>`${esc(k)} ${num(v)}개`).join(" · ") || "표시된 결함 없음"}</span></div></div><span class="status ${j.status}">${labels[j.status]}</span></div><div class="image-toolbar"><span>원본 영상 · SHA 확인 후 제공</span><label><input id="bbox-toggle" type="checkbox" checked> 주석 박스</label></div><div class="image-view"><div class="image-inner" id="image-inner"><img id="sample-image" alt="${esc(item.id)} 용접 train/val 영상"><svg id="boxes" aria-label="고정 원본 결함 주석"></svg></div></div><p class="image-caption">박스는 원본 주석이며 모델 예측이 아닙니다. 좌표 단위: 원본 화소 · 치수 합부 판정 없음</p><div class="text-block"><h4>QWEN · 생성된 설명 <span>${num(item.output_tokens)} tokens / ${esc(item.finish_reason)}</span></h4><p>${esc(item.text || "생성 결과 없음")}</p></div><div class="text-block"><h4>PHI · 자동 판정 <span>학습 품질 승인 아님</span></h4><div class="judgment-box">${j.status!=="pending" ? `<div class="judgment-flags">${["grounded","scope_ok","language_ok"].map(k=>`<span>${k} <b>${j[k]===true?"true":j[k]===false?"false":"—"}</b></span>`).join("")}</div>`:""}<p>${esc(j.reason)}</p>${j.text ? `<details><summary>판정 원문 보기</summary><pre>${esc(j.text)}</pre></details>`:""}</div></div>${Object.keys(item.defects).length ? `<div class="small-warning"><strong>주석이 요구한 한계 설명</strong><br>${esc(item.evidence.limitation)}<br>생성 문장이 이를 충족하는지는 별도 검토 대상입니다.</div>` : `<p class="footnote">무결함 행에는 “표시된 결함 없음”의 짧은 문장 예외가 적용됩니다. 자동 판정 과탈락 가능성을 검토해야 합니다.</p>`}`;
    document.querySelector(".image-toolbar > span").textContent = "고정 입력 영상 · SHA 확인 후 제공";
    document.querySelector(".image-caption").textContent = "히스토그램 매칭 전처리 타일입니다. 박스는 고정 주석이며 모델 예측이 아닙니다. 좌표: 입력 화소 · 치수 합부 판정 없음";
    const img=$("sample-image"),svg=$("boxes");
    img.addEventListener("load",()=>{svg.setAttribute("viewBox",`0 0 ${img.naturalWidth} ${img.naturalHeight}`);svg.innerHTML=item.boxes.filter(b=>Array.isArray(b.bbox_px)&&b.bbox_px.length===4&&b.bbox_px.every(Number.isFinite)).map(b=>{const [x1,y1,x2,y2]=b.bbox_px;return `<rect x="${x1}" y="${y1}" width="${Math.max(0,x2-x1)}" height="${Math.max(0,y2-y1)}" fill="#e6bb5222" stroke="#f1c366" stroke-width="2" vector-effect="non-scaling-stroke"/>`;}).join("");});
    img.addEventListener("error",()=>{$("image-inner").innerHTML='<div class="image-error">영상을 제공할 수 없습니다.<br>로컬 파일 부재, 안전 경로 제한 또는 SHA 불일치입니다.</div>';});
    img.src=`/api/image?id=${encodeURIComponent(item.id)}`;
    $("bbox-toggle").addEventListener("change",e=>{if($("boxes"))$("boxes").style.display=e.target.checked?"block":"none";});
  }catch(e){error(e.message);}
}
function eventCounts(item) {
  if (item.stage === "generate" && item.generated_shards != null) return `생성 ${num(item.generated_shards)}`;
  if (item.stage === "judge" && item.judged_shards != null) return `판정 ${num(item.judged_shards)}`;
  if (item.generated_shards != null && item.judged_shards != null) return `생성 ${num(item.generated_shards)} / 판정 ${num(item.judged_shards)}`;
  if (item.judged_shards != null) return `판정 ${num(item.judged_shards)}`;
  if (item.generated_shards != null) return `생성 ${num(item.generated_shards)}`;
  return "—";
}
async function loadEvents(){const data=await api("/api/events");$("events-table").innerHTML=data.items.slice().reverse().map(x=>`<tr><td>${esc(time(x.at))}</td><td>${esc(x.event)}</td><td>${esc(x.stage||"—")}</td><td>${esc(eventCounts(x))}</td><td>${x.external_rc==null?"—":num(x.external_rc)}</td><td>${esc(x.reason || (x.projected_won?`전량 예상 ${num(x.projected_won)}원`:x.message||""))}</td></tr>`).join("");}
function markDisconnected(message) {
  error(message);
  $("connection").textContent = "연결 오류 · 마지막 관측값";
  $("connection").classList.remove("live");
  $("freshness").classList.add("stale");
  $("freshness").textContent = "자료 연결에 실패했습니다. 표시된 숫자는 마지막 수신값이며, 현재 작업 상태가 아닙니다. 새로고침하여 다시 확인하세요.";
  $("gpu").textContent = "현재 상태 확인 불가";
  $("pipeline").querySelectorAll(".current").forEach(node => { node.classList.remove("current"); node.querySelector("small").textContent = "현재 상태 확인 불가"; });
}
async function refresh(initial=false){$("refresh").disabled=true;try{renderSummary(await api("/api/summary"));await loadEvents();if(initial)await loadSamples();error("");}catch(e){markDisconnected(e.message);}finally{$("refresh").disabled=false;}}
$("refresh").addEventListener("click",()=>refresh());
for(const key of ["split","client","defect","judge"])$(key).addEventListener("change",()=>loadSamples());
let searchTimer;$("search").addEventListener("input",()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>loadSamples(),280);});
$("prev").addEventListener("click",()=>{offset=Math.max(0,offset-12);loadSamples(false);});$("next").addEventListener("click",()=>{offset+=12;loadSamples(false);});
const sections=[...document.querySelectorAll("main>section")];if("IntersectionObserver"in window){const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){document.querySelectorAll("nav a").forEach(a=>a.classList.toggle("active",a.hash===`#${entry.target.id}`));}},{rootMargin:"-15% 0px -60% 0px"});sections.forEach(s=>observer.observe(s));}
refresh(true);setInterval(()=>refresh(),30000);
