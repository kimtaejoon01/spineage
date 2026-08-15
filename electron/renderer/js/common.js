// electron/renderer/js/common.js
export const API = 'http://127.0.0.1:8000';

export const $ = (id) => document.getElementById(id);
export const show = (id, on) => $(id).classList.toggle('d-none', !on);

/** POST helper with optional spinner-id */
export async function post(url, formData, spinId) {
  if (spinId) show(spinId, true);
  try {
    const res = await fetch(API + url, { method: 'POST', body: formData });
    const contentType = res.headers.get('content-type') || '';
    const payload = contentType.includes('application/json')
      ? await res.json()
      : await res.text();

    if (!res.ok) {
      const detail = typeof payload === 'object' && payload?.detail
        ? payload.detail
        : String(payload || `HTTP ${res.status}`);
      throw new Error(detail);
    }
    return payload;
  } finally {
    if (spinId) show(spinId, false);
  }
}
