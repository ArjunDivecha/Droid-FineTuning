# Electron Loading Fix

## Problem
Electron app was failing to start with error:
```
TypeError: Cannot read properties of undefined (reading 'whenReady')
```

## Root Cause
The `ELECTRON_RUN_AS_NODE=1` environment variable was set in the development environment (specifically in Claude Code/VS Code contexts). When this variable is set, Electron runs as plain Node.js instead of as the Electron runtime, which causes `require('electron')` to return the npm package wrapper (a string path) instead of the actual Electron API module.

## Solution
The `startmlxnew` shell command now unsets this variable before launching:

```bash
unset ELECTRON_RUN_AS_NODE
```

## Technical Details
- When `ELECTRON_RUN_AS_NODE=1` is set, Electron's module interception doesn't work
- `require('electron')` resolves to `node_modules/electron/index.js` which exports a file path string
- Without the variable, Electron properly intercepts the require and provides its runtime APIs
- This issue affects any Electron app run from environments that set this variable (like VS Code's integrated terminal)

## How to Test
Run from your native terminal:
```bash
startmlxnew
```

The Electron window should now open successfully and you should see:
```
🪟 Creating main window...
✅ BrowserWindow created
Window ready-to-show event fired
Window shown and focused
```

## Date
Fixed: November 1, 2024
