//preload.js
const { contextBridge } = require('electron');
contextBridge.exposeInMainWorld('api', { port: 8000 });
