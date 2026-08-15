// electron/renderer/js/h5.js
import { API, $, post } from './common.js';

let session = null;           // 세션 ID (h5 or preproc)
let items   = [];             // [{fid,name,preview:{img,muscle,vertebra}}]
let selectedFids = new Set(); // 3개 선택
let focusFid = null;          // 미리보기용 포커스 파일
let source = null;            // 'h5' | 'preproc'
let volume = null;            // 현재 포커스의 전체 Base64 슬라이스

const cvs = {
  img : $('cv-img').getContext('2d'),
  mus : $('cv-mus').getContext('2d'),
  ver : $('cv-ver').getContext('2d'),
};

function showErr(id, msg){ $(id).textContent = msg || ''; }
function toggleSpin(id, on){ $(id).classList.toggle('d-none', !on); }

function drawSlice(z){
  if(!volume) return;
  $('zlab').textContent = z;

  const drawBase = (ctx, b64) => new Promise(res=>{
    const img = new Image(); img.src = 'data:image/png;base64,'+b64;
    img.onload=()=>{ ctx.clearRect(0,0,256,256); ctx.drawImage(img,0,0,256,256); res(); };
  });
  const drawMaskTint = (ctx, b64, color) => new Promise(res=>{
    const m = new Image(); m.src = 'data:image/png;base64,'+b64;
    m.onload=()=>{
      // mask를 alpha로 쓰고 색 채우기
      const tmp = document.createElement('canvas'); tmp.width=256; tmp.height=256;
      const tctx = tmp.getContext('2d');
      tctx.drawImage(m,0,0,256,256);                    // grayscale mask
      const mask = tctx.getImageData(0,0,256,256);
      const imgd = ctx.getImageData(0,0,256,256);
      // color = [r,g,b], alpha 0.35
      const [r,g,b] = color;
      for(let i=0;i<mask.data.length;i+=4){
        const v = mask.data[i]; // 0~255
        if(v>0){
          imgd.data[i  ] = Math.max(imgd.data[i  ], r);
          imgd.data[i+1] = Math.max(imgd.data[i+1], g);
          imgd.data[i+2] = Math.max(imgd.data[i+2], b);
        }
      }
      ctx.putImageData(imgd,0,0);
      ctx.globalAlpha=0.35; ctx.drawImage(m,0,0,256,256); ctx.globalAlpha=1;
      res();
    };
  });

  // 원본
  drawBase(cvs.img, volume.img[z]).then(()=>{});
  // 원본 + 근육(빨강)
  drawBase(cvs.mus, volume.img[z]).then(()=>drawMaskTint(cvs.mus, volume.muscle[z], [255,0,0]));
  // 원본 + 척추(초록)
  drawBase(cvs.ver, volume.img[z]).then(()=>drawMaskTint(cvs.ver, volume.vertebra[z], [0,0,255]));
}



$('z').oninput = e => drawSlice(+e.target.value);

function renderList(){
  const list = $('list');
  list.innerHTML = '';
  items.forEach(it=>{
    const wrap = document.createElement('div');
    wrap.className = 'card-thumb';
    wrap.innerHTML = `
      <div class="card p-2">
        <img class="thumb-img" src="data:image/png;base64,${it.preview.img}">
        <div class="small text-truncate mt-1" title="${it.name}">${it.name}</div>
        <div class="d-flex justify-content-between align-items-center mt-1">
          <div class="form-check">
            <input class="form-check-input" type="checkbox" id="chk-${it.fid}">
            <label class="form-check-label small" for="chk-${it.fid}">선택</label>
          </div>
          <button class="btn btn-sm btn-outline-secondary" data-view="${it.fid}">보기</button>
        </div>
      </div>
    `;
    // 체크박스
    wrap.querySelector(`#chk-${it.fid}`).checked = selectedFids.has(it.fid);
    wrap.querySelector(`#chk-${it.fid}`).onchange = (e)=>{
      if(e.target.checked){
        if(selectedFids.size >= 3){ e.target.checked=false; return; }
        selectedFids.add(it.fid);
      }else{
        selectedFids.delete(it.fid);
      }
      $('btn-predict').disabled = !(session && selectedFids.size===3);
    };
    // 보기 버튼
    wrap.querySelector(`[data-view="${it.fid}"]`).onclick = async ()=>{
      try{
        focusFid = it.fid;
        const q = new URLSearchParams({session, fid: focusFid});
        const endpoint = (source==='preproc') ? '/preproc/volume' : '/h5_volume';
        const res = await fetch(`${API}${endpoint}?${q.toString()}`);
        if(!res.ok){
          const txt = await res.text(); throw new Error(`Preview failed: ${res.status} ${txt}`);
        }
        const jsn = await res.json();
        volume = jsn.volume;
        $('z').value = 12;
        drawSlice(12);
      }catch(err){
        console.error(err);
        alert('미리보기 로드 실패\n' + err.message);
      }
    };
    list.appendChild(wrap);
  });
}

/* ───────── 1A. NIfTI 전처리 ───────── */
$('btn-nifti').onclick = async ()=>{
  showErr('err-nifti','');
  const f = $('nifti').files[0];
  if(!f){ showErr('err-nifti','NIfTI 파일을 선택하세요'); return; }

  const fd = new FormData();
  fd.append('file', f);
  const gid = $('gpu_nifti').value.trim();
  if(gid) fd.append('gpu_id', gid);
  const name = f.name.toLowerCase();
  fd.append('file_type', name.endsWith('.nii') ? '.nii' : '.nii.gz');

  toggleSpin('spin-nifti', true);
  try{
    const res = await fetch(`${API}/preproc/nifti`, {method:'POST', body:fd});
    if(!res.ok){
      const txt = await res.text();
      throw new Error(`전처리 실패 (NIfTI): ${res.status}\n${txt}`);
    }
    const jsn = await res.json();
    session = jsn.session; items = jsn.items; source='preproc';
    // 근육 사용 가능 여부에 따라 옵션 제어
    if (jsn.muscle_available === false) {
      $('bone_only').value = 'bone_only';
      const opt = [...$('bone_only').options].find(o=>o.value==='bone_muscle');
      if (opt) opt.disabled = true;
    } else {
      const opt = [...$('bone_only').options].find(o=>o.value==='bone_muscle');
      if (opt) opt.disabled = false;
    }
    selectedFids.clear(); focusFid=null; volume=null;
    $('btn-predict').disabled = !(session && selectedFids.size===3);
    renderList();
  }catch(err){
    console.error(err); showErr('err-nifti', err.message);
    alert('NIfTI 전처리 실패\n' + err.message);
  }finally{
    toggleSpin('spin-nifti', false);
  }
};

/* ───────── 1B. DICOM 전처리 ───────── */
$('btn-dicom').onclick = async ()=>{
  showErr('err-dicom','');
  const files = $('dicoms').files;
  if(!files.length){ showErr('err-dicom','DICOM 파일들을 선택하세요'); return; }

  const fd = new FormData();
  [...files].forEach(f=>fd.append('files', f));
  const gid = $('gpu_dicom').value.trim();
  if(gid) fd.append('gpu_id', gid);
  fd.append('file_type', '.nii.gz');

  toggleSpin('spin-dicom', true);
  try{
    const res = await fetch(`${API}/preproc/dicom`, {method:'POST', body:fd});
    if(!res.ok){
      const txt = await res.text();
      throw new Error(`전처리 실패 (DICOM): ${res.status}\n${txt}`);
    }
    const jsn = await res.json();
    session = jsn.session; items = jsn.items; source='preproc';
    if (jsn.muscle_available === false) {
      $('bone_only').value = 'bone_only';
      const opt = [...$('bone_only').options].find(o=>o.value==='bone_muscle');
      if (opt) opt.disabled = true;
    } else {
      const opt = [...$('bone_only').options].find(o=>o.value==='bone_muscle');
      if (opt) opt.disabled = false;
    }
    selectedFids.clear(); focusFid=null; volume=null;
    $('btn-predict').disabled = !(session && selectedFids.size===3);
    renderList();
  }catch(err){
    console.error(err); showErr('err-dicom', err.message);
    alert('DICOM 전처리 실패\n' + err.message);
  }finally{
    toggleSpin('spin-dicom', false);
  }
};

/* ───────── 1C. HDF5 직접 업로드 ───────── */
$('btn-load').onclick = async ()=>{
  showErr('err-h5','');
  const files = $('files').files;
  if(!files.length){ showErr('err-h5','HDF5 파일들을 선택하세요'); return; }
  try{
    const fd = new FormData();
    [...files].forEach(f=>fd.append('files', f));
    const {session:sid, items:its} = await post('/load_h5_multi', fd, 'spin-load');
    session = sid; items = its; source='h5';
    selectedFids.clear(); focusFid=null; volume=null;
    // HDF5 직접 업로드는 muscle_available 정보를 모르므로 옵션은 그대로(사용자 선택)
    $('btn-predict').disabled = !(session && selectedFids.size===3);
    renderList();
  }catch(err){
    console.error(err); showErr('err-h5', String(err));
  }
};

/* ───────── 예측 ───────── */
$('btn-predict').onclick = async ()=>{
  if(selectedFids.size !== 3){ alert('척추체 파일 3개를 선택하세요'); return; }

  const hidden_dim   = 256;
  const classifier   = $('classifier').value;
  const dimension    = $('dimension').value;
  const channelsnum  = parseInt($('channels_num_2d').value, 10);
  let bone_only      = $('bone_only').value;

  // 근육 옵션 비활성화 상태에서 사용자가 강제로 선택했다면 보호장치
  const muscleOpt = [...$('bone_only').options].find(o=>o.value==='bone_muscle');
  if (muscleOpt && muscleOpt.disabled && bone_only === 'bone_muscle') {
    alert("v7_2d_unet이 없어 근육 마스크를 사용할 수 없습니다. 'bone_only'로 진행합니다.");
    $('bone_only').value = 'bone_only';
    bone_only = 'bone_only';
  }

  const args = {
    dimension, pretrained: true,
    classifier,
    hidden_dim,
    dim_in: hidden_dim * 9,     // 9스택 전제
    max_followup: 10,
    heads_balance: 0,
    bn_order: 'bn_relu',
    task: 'MTL',
    cls_enabled: $('enable-cls').checked,
    pred_enabled: $('enable-pred').checked,
    bone_only,
    channels_num_2d: channelsnum,
  };

  const fd = new FormData();
  fd.append('session', session);
  fd.append('selected', Array.from(selectedFids).join(','));
  fd.append('args_json', JSON.stringify(args));

  const endpoint = (source==='preproc') ? '/preproc/predict' : '/predict_h5_multi';
  try{
    const res = await post(endpoint, fd, 'spin-predict');
    // CLS
    if (typeof res.cls_probability === 'number') {
      const p = res.cls_probability;
      $('result-cls').innerHTML =
        `<div class="alert ${p>0.5?'alert-danger':'alert-success'} mb-2">
           분류(양성) 확률: <strong>${(p*100).toFixed(1)}%</strong>
           <div class="small text-muted mt-1">선택: ${(res.selected_names||[]).join(' · ')}</div>
         </div>`;
    } else {
      $('result-cls').innerHTML = '';
    }
    // PRED
    if (Array.isArray(res.pred_curve)) {
      const cells = res.pred_curve.map((v,i)=>
        `<tr><td>${i+1}</td><td>${(v*100).toFixed(1)}%</td></tr>`).join('');
      $('result-pred').innerHTML =
        `<div class="card">
           <div class="card-header py-1">위험도/누적 확률 (T=${res.pred_curve.length})</div>
           <div class="card-body p-2">
             <table class="table table-sm mb-0">
               <thead><tr><th>시점</th><th>확률</th></tr></thead>
               <tbody>${cells}</tbody>
             </table>
           </div>
         </div>`;
    } else {
      $('result-pred').innerHTML = '';
    }
  }catch(err){
    console.error(err);
    alert('예측 실패\n' + err.message);
  }
};

// 초기 상태
renderList();
