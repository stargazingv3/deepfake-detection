chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === 'classifyImage') {
        fetch('http://localhost:5000/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                imageBase64: request.imageBase64
            }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Deepfake result:', data);
            // Send the result to the popup
            chrome.runtime.sendMessage({
                action: 'showResult',
                prediction: data.prediction,
                confidence: data.confidence
            });
        })
        .catch(error => console.error('Error:', error));
    }
});
