// electron/renderer/js/common.js
export const API = 'http://127.0.0.1:8000';

export const $    = (id)=>document.getElementById(id);
export const show = (id,on)=>$(id).classList.toggle('d-none',!on);

/** POST helper with optional spinner-id */
export async function post(url, formData, spinId){
  if(spinId) show(spinId,true);
  const res = await fetch(API+url,{method:'POST',body:formData});
  if(spinId) show(spinId,false);
  return res.json();
}
