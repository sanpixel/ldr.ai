// Content script for LDR Auto Print extension
// Runs on ldr.clocknumbers.com pages

// Listen for custom events from the web page
window.addEventListener('LDR_PDF_READY', (event) => {
  const pdfData = event.detail.pdfData;
  
  // Send message to background script to handle printing
  chrome.runtime.sendMessage({
    action: 'printPDF',
    pdfData: pdfData
  }, (response) => {
    if (response && response.printed) {
      console.log('LDR Auto Print: PDF sent to printer');
    } else if (response && !response.printed) {
      console.log('LDR Auto Print: Ready mode is off, print skipped');
    }
  });
});

// Notify page that extension is loaded
window.postMessage({ type: 'LDR_EXTENSION_LOADED' }, '*');
