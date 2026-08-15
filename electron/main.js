// main.js
const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');
const isDev = process.defaultApp || /--dev/.test(process.argv[2] || '');

let pyProc = null;
const PORT = 8000;
const baseDir = isDev ? path.join(__dirname, '..') : process.resourcesPath;

function startPython() {
  const python = process.platform === 'win32'
    ? path.join(baseDir, 'venv', 'Scripts', 'python.exe')
    : path.join(baseDir, 'venv', 'bin', 'python');
  const backend = path.join(baseDir, 'backend', 'server.py');

  pyProc = spawn(python, [backend], {
    cwd: baseDir,
    stdio: 'inherit'
  });

  pyProc.on('error', (err) => {
    console.error('Failed to start Python:', err);
    dialog.showErrorBox('Backend start failed', String(err));
    app.quit();
  });

  pyProc.on('close', (code) => {
    console.log('Python exited', code);
    if (!app.isQuiting) app.quit();
  });
}

function waitForBackend(timeoutMs = 120000) {
  const startedAt = Date.now();
  return new Promise((resolve, reject) => {
    const check = () => {
      const req = http.get(`http://127.0.0.1:${PORT}/health`, (res) => {
        res.resume();
        if (res.statusCode === 200) {
          resolve();
          return;
        }
        retry();
      });
      req.on('error', retry);
      req.setTimeout(1500, () => req.destroy());
    };

    const retry = () => {
      if (Date.now() - startedAt >= timeoutMs) {
        reject(new Error('Python backend did not become ready in time.'));
        return;
      }
      setTimeout(check, 500);
    };

    check();
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1600,
    height: 1000,
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  win.once('ready-to-show', () => win.show());
}

app.whenReady().then(async () => {
  startPython();
  try {
    await waitForBackend();
    createWindow();
  } catch (err) {
    console.error(err);
    dialog.showErrorBox('Backend unavailable', String(err));
    app.quit();
  }
});

app.on('before-quit', () => {
  app.isQuiting = true;
  if (pyProc) pyProc.kill();
});
