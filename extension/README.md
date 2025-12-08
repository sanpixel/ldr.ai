# LDR Auto Print Chrome Extension

Auto-print highlighted PDF previews from Legal Description Reader.

## Installation

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `extension` folder
5. The extension icon will appear in your toolbar

## Usage

1. Click the extension icon to open the popup
2. Toggle "Ready Mode" ON to enable auto-printing
3. Visit https://ldr.clocknumbers.com
4. Process a PDF - when complete, it will auto-print if Ready Mode is ON
5. Toggle Ready Mode OFF to disable auto-printing

## Icons

The extension needs icon files. Create simple 16x16, 48x48, and 128x128 PNG images named:
- `icon16.png`
- `icon48.png`
- `icon128.png`

Place them in the `extension` folder.

## How It Works

1. Content script (`content.js`) runs on ldr.clocknumbers.com
2. Web page dispatches `LDR_PDF_READY` event when PDF is ready
3. Content script forwards to background service worker
4. Background worker checks if Ready Mode is enabled
5. If enabled, opens PDF in hidden window and triggers print dialog
6. Print dialog appears for user to confirm/cancel

## Notes

- Print dialog still appears (browser security requirement)
- Extension only works on ldr.clocknumbers.com
- Ready Mode state persists across browser sessions
