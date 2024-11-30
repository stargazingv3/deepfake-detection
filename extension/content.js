chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'detectImage') {
        console.log("Deepfake Detector: Received detect image request");
        console.log("Image URL:", request.imageUrl);

        // Function to convert file to base64
        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => resolve(reader.result);
                reader.onerror = error => reject(error);
            });
        }

        // Try to fetch the file
        fetch(request.imageUrl)
            .then(response => response.blob())
            .then(blob => {
                console.log("Deepfake Detector: Blob retrieved successfully");
                
                // Convert blob to base64
                return fileToBase64(blob);
            })
            .then(base64Image => {
                console.log("Deepfake Detector: Image converted to base64");

                // Send for classification
                chrome.runtime.sendMessage({ 
                    action: 'classifyImage', 
                    imageBase64: base64Image
                }, function(response) {
                    console.log("Deepfake Detector: Classification response:", response);
                    createResultPopup(response);
                });
            })
            .catch(error => {
                console.error("Deepfake Detector: Error processing image", error);
                createResultPopup({
                    prediction: 'error',
                    confidence: 0,
                    errorMessage: 'Failed to load image: ' + error.message
                });
            });
    }
});

// Create result popup
function createResultPopup(result) {
    console.log("Deepfake Detector: Creating result popup");
    
    // Create popup div
    const resultDiv = document.createElement('div');
    resultDiv.style.position = 'fixed';
    resultDiv.style.top = '50%';
    resultDiv.style.left = '50%';
    resultDiv.style.transform = 'translate(-50%, -50%)';
    resultDiv.style.background = 'white';
    resultDiv.style.padding = '20px';
    resultDiv.style.border = '2px solid black';
    resultDiv.style.zIndex = '10000';
    resultDiv.style.boxShadow = '0 0 10px rgba(0,0,0,0.5)';
    resultDiv.style.textAlign = 'center';
    resultDiv.style.maxWidth = '300px';
    
    // Determine result styling
    const prediction = result.prediction || 'error';
    const probability = result.probability !== undefined ? result.probability : 0; // Use probability instead of confidence
    
    resultDiv.innerHTML = `
        <h2>Deepfake Detection Result</h2>
        ${result.errorMessage ? 
            `<p style="color: red;">${result.errorMessage}</p>` :
            `<p><strong>Prediction:</strong> ${prediction.toUpperCase()}</p>
             <p><strong>Probability:</strong> ${(probability * 100).toFixed(2)}%</p>`
        }
        <button id="close-result" style="margin-top: 10px; padding: 5px 10px;">Close</button>
    `;
    
    // Style based on prediction
    if (prediction === 'fake') {
        resultDiv.style.backgroundColor = '#FFE6E6';  // Light red
        resultDiv.style.color = '#8B0000';  // Dark red text
        resultDiv.style.border = '2px solid #FF0000';  // Red border
    } else if (prediction === 'real') {
        resultDiv.style.backgroundColor = '#E6FFE6';  // Light green
        resultDiv.style.color = '#006400';  // Dark green text
        resultDiv.style.border = '2px solid #00FF00';  // Green border
    } else {
        resultDiv.style.backgroundColor = '#F0F0F0';  // Neutral gray
        resultDiv.style.color = '#333333';  // Dark gray text
        resultDiv.style.border = '2px solid #CCCCCC';  // Gray border
    }
    
    // Add to document
    document.body.appendChild(resultDiv);
    
    // Add close button functionality
    document.getElementById('close-result').addEventListener('click', () => {
        document.body.removeChild(resultDiv);
    });
}
