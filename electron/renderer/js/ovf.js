// electron/renderer/js/ovf.js
import {API, $, show, post} from './common.js';

let session=null, thumbs=[], selected=[], kps=[];
const canvas=new fabric.Canvas('cv',{selection:false});

function toggleSel(idx,img){
  if(selected.includes(idx)){selected=selected.filter(x=>x!==idx);img.classList.remove('sel');}
  else{if(selected.length>=3)return;selected.push(idx);img.classList.add('sel');}
  selected.sort((a,b)=>a-b);
  if(selected.length===3){
    fabric.Image.fromURL(thumbs[selected[1]],o=>{
      canvas.setBackgroundImage(o,canvas.renderAll.bind(canvas));
      kps.length=0; drawPoints(); refreshTable(); checkReady();
    });
  }
  checkReady();
}

function drawPoints(){
  canvas.getObjects('circle').forEach(o=>canvas.remove(o));
  kps.forEach(p=>canvas.add(new fabric.Circle({radius:4,left:p.x-4,top:p.y-4,fill:'red',stroke:'yellow',strokeWidth:1,selectable:false})));
}
function refreshTable(){
  const tb=$('kptable'); tb.innerHTML='';
  for(let i=0;i<6;i++){
    const tr=tb.insertRow(); tr.insertCell().textContent=i+1;
    ['x','y'].forEach(c=>{
      tr.insertCell().innerHTML=`<input data-i="${i}" data-c="${c}" value="${kps[i]?.[c]??''}" class="form-control form-control-sm" style="width:70px">`;
    });
    tr.insertCell().innerHTML=`<button class="btn btn-link btn-sm text-danger" data-del="${i}">&times;</button>`;
  }
}
canvas.on('mouse:down',e=>{
  if(kps.length>=6||!e.pointer)return;
  kps.push({x:Math.round(e.pointer.x),y:Math.round(e.pointer.y)});
  drawPoints();refreshTable();checkReady();
});
$('kptable').addEventListener('input',e=>{
  if(!e.target.dataset.c)return;
  const i=+e.target.dataset.i,c=e.target.dataset.c;
  if(!kps[i])kps[i]={x:0,y:0}; kps[i][c]=+e.target.value;
  drawPoints(); checkReady();
});
$('kptable').addEventListener('click',e=>{
  if(!e.target.dataset.del)return;
  kps.splice(+e.target.dataset.del,1); drawPoints(); refreshTable(); checkReady();
});
$('btn-reset').onclick=()=>{kps.length=0; drawPoints(); refreshTable(); checkReady();};

$('use_clin').onchange=e=> $('clin').style.display=e.target.checked?'block':'none';

async function checkReady(){
  const ready = session && selected.length===3 && kps.length===6;
  $('btn-predict').disabled=!ready;
  if(ready) {
    try {
      await previewROI();
    } catch (err) {
      console.error(err);
      $('roi').classList.add('d-none');
      const placeholder = $('roi-placeholder');
      if(placeholder) placeholder.classList.remove('d-none');
    }
  } else {
    $('roi').classList.add('d-none');
    const placeholder = $('roi-placeholder');
    if(placeholder) placeholder.classList.remove('d-none');
  }
}

async function previewROI(){
  const fd=new FormData();
  fd.append('session',session);
  fd.append('frames',selected.join(','));
  fd.append('kps',kps.map(p=>`${p.y},${p.x}`).join(','));
  const {png}=await post('/preview', fd, 'spin-roi');
  $('roi').src='data:image/png;base64,'+png;
  $('roi').classList.remove('d-none');
  const placeholder = $('roi-placeholder');
  if(placeholder) placeholder.classList.add('d-none');
}

$('btn-load').onclick=async()=>{
  const files=$('files').files;
  if(!files.length){alert('파일 선택');return;}
  const fd=new FormData();
  [...files].forEach(f=>fd.append('files',f));

  try {
    const {session:sid,thumbs:ths}=await post('/load', fd, 'spin-load');

    session=sid; thumbs=ths; selected.length=0; kps.length=0;
    drawPoints(); refreshTable();
    $('roi').classList.add('d-none');
    const placeholder = $('roi-placeholder');
    if(placeholder) placeholder.classList.remove('d-none');

    const cont=$('thumbs'); cont.innerHTML='';
    thumbs.forEach((b64,i)=>{
      const im=new Image(); im.src=b64; im.className='thumb'; im.title=i;
      im.onclick=()=>toggleSel(i,im); cont.appendChild(im);
    });

    if(window.removeUploadLoading) window.removeUploadLoading();
  } catch (err) {
    console.error(err);
    if(window.removeUploadLoading) window.removeUploadLoading();
    alert('DICOM 로드 실패\n' + err.message);
  }
};

$('btn-predict').onclick=async()=>{
  const fd=new FormData();
  fd.append('session',session);
  fd.append('frames',selected.join(','));
  fd.append('kps',kps.map(p=>`${p.y},${p.x}`).join(','));
  const use=$('use_clin').checked; fd.append('use_clinical',use);
  if(use){
    ['age','sex','bmd'].forEach(id=>fd.append(id,$(id).value));
    ['pre','post'].forEach(id=>fd.append(id,$(id).checked));
  }

  try {
    const {probability:p}=await post('/predict', fd, 'spin-predict');
    $('result').innerHTML=`<div class="alert ${p>0.5?'alert-danger':'alert-success'}">
        골다공성 압박골절 진행 확률: <strong>${(p*100).toFixed(1)}%</strong>
      </div>`;
  } catch (err) {
    console.error(err);
    alert('예측 실패\n' + err.message);
  }
};
