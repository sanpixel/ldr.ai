// Popup script for LDR Auto Print extension

const toggle = document.getElementById('readyToggle');
const status = document.getElementById('status');

// Load current ready mode state
chrome.storage.local.get(['readyMode'], (result) => {
  const isReady = result.readyMode === true;
  toggle.checked = isReady;
  updateStatus(isReady);
});

// Handle toggle changes
toggle.addEventListener('change', (event) => {
  const isReady = event.target.checked;
  chrome.storage.local.set({ readyMode: isReady }, () => {
    updateStatus(isReady);
  });
});

function updateStatus(isReady) {
  if (isReady) {
    status.textContent = 'Ready mode is ON - PDFs will auto-print';
    status.classList.add('ready');
  } else {
    status.textContent = 'Ready mode is OFF';
    status.classList.remove('ready');
  }
}
