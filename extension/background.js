// Create context menu when extension is installed
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "deepfake-detector",
        title: "Detect Deepfake",
        contexts: ["image"]
    });
    console.log("Deepfake Detector: Context menu created");
});

// Listen for context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "deepfake-detector") {
        console.log("Deepfake Detector: Context menu clicked");
        console.log("Image URL:", info.srcUrl);
        
        // Send message to content script with image URL
        chrome.tabs.sendMessage(tab.id, {
            action: 'detectImage',
            imageUrl: info.srcUrl
        }, (response) => {
            if (chrome.runtime.lastError) {
                console.error("Error sending message:", chrome.runtime.lastError);
            }
        });
    }
});

// Handle image classification request
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'classifyImage') {
        console.log("Deepfake Detector: Received image for classification");
        
        fetch('http://localhost:5000/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ imageBase64: request.imageBase64 }),
        })
        .then(response => response.json())
        .then(data => {
            console.log("Deepfake Detector: Classification result:", data);
            sendResponse(data);  // Send the classification result with 'probability'
        })
        .catch(error => {
            console.error('Deepfake Detector: Classification error:', error);
            sendResponse({ prediction: 'error', probability: 0 });
        });
        
        // Indicate that we wish to send a response asynchronously
        return true;
    }
});
