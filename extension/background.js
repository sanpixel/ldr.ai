// Background service worker for LDR Auto Print extension

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'printPDF') {
    // Check if ready mode is enabled
    chrome.storage.local.get(['readyMode'], (result) => {
      if (result.readyMode === true) {
        // Ready mode is on - proceed with printing
        printPDF(request.pdfData, sender.tab.id);
        sendResponse({ success: true, printed: true });
      } else {
        // Ready mode is off - ignore request
        sendResponse({ success: true, printed: false, reason: 'Ready mode is off' });
      }
    });
    return true; // Keep message channel open for async response
  }
});

function printPDF(pdfDataUrl, tabId) {
  // Create a new window with the PDF
  chrome.windows.create({
    url: pdfDataUrl,
    type: 'popup',
    focused: false,
    width: 1,
    height: 1
  }, (window) => {
    // Wait for the window to load, then print
    chrome.tabs.onUpdated.addListener(function listener(updatedTabId, info) {
      if (updatedTabId === window.tabs[0].id && info.status === 'complete') {
        chrome.tabs.onUpdated.removeListener(listener);
        
        // Trigger print
        chrome.scripting.executeScript({
          target: { tabId: window.tabs[0].id },
          func: () => {
            window.print();
            // Close window after print dialog
            setTimeout(() => window.close(), 1000);
          }
        });
      }
    });
  });
}

// Initialize ready mode to false on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({ readyMode: false });
});
