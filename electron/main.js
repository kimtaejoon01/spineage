// main.js
const { app, BrowserWindow } = require('electron');
const path   = require('path');
const { spawn } = require('child_process');
const isDev = process.defaultApp || /--dev/.test(process.argv[2] || '');

let pyProc = null;
const PORT  = 8000;

// dev → <repo>,  prod → <App>/Contents/Resources
const baseDir = isDev ? path.join(__dirname, '..') : process.resourcesPath;

function startPython () {
  const python  = process.platform === 'win32'
    ? path.join(baseDir, 'venv', 'Scripts', 'python.exe')
    : path.join(baseDir, 'venv', 'bin',    'python');

  const backend = path.join(baseDir, 'backend', 'server.py');

  pyProc = spawn(python, [backend, PORT], {
    cwd: baseDir,
    stdio: 'inherit'
  });

  pyProc.on('close', code => {
    console.log('Python exited', code);
    if (!app.isQuiting) app.quit();
  });
}

function createWindow () {
  const win = new BrowserWindow({
    width: 1600,
    height: 1000,
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    }
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

app.whenReady().then(() => {
  startPython();
  createWindow();
});

app.on('before-quit', () => {
  app.isQuiting = true;
  if (pyProc) pyProc.kill();
});
