import { contextBridge, ipcRenderer, clipboard } from 'electron';

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // File dialogs
  showOpenDialog: (options: any) => ipcRenderer.invoke('show-open-dialog', options),
  showSaveDialog: (options: any) => ipcRenderer.invoke('show-save-dialog', options),
  selectDirectory: () => ipcRenderer.invoke('show-open-dialog', { 
    properties: ['openDirectory'] 
  }),
  
  // Menu events
  onMenuOpenTrainingData: (callback: () => void) => 
    ipcRenderer.on('menu-open-training-data', callback),
  onMenuStartTraining: (callback: () => void) => 
    ipcRenderer.on('menu-start-training', callback),
  onMenuStopTraining: (callback: () => void) => 
    ipcRenderer.on('menu-stop-training', callback),
  
  // Platform info
  platform: process.platform,

  // Clipboard operations
  clipboard: {
    writeText: (text: string) => clipboard.writeText(text),
    readText: () => clipboard.readText(),
    clear: () => clipboard.clear()
  },
});

// Type definitions for the exposed API
declare global {
  interface Window {
    electronAPI: {
      showOpenDialog: (options: any) => Promise<any>;
      showSaveDialog: (options: any) => Promise<any>;
      selectDirectory: () => Promise<any>;
      onMenuOpenTrainingData: (callback: () => void) => void;
      onMenuStartTraining: (callback: () => void) => void;
      onMenuStopTraining: (callback: () => void) => void;
      platform: string;
      clipboard: {
        writeText: (text: string) => void;
        readText: () => string;
        clear: () => void;
      };
    };
  }
}