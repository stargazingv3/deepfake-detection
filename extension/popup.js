document.addEventListener('DOMContentLoaded', function() {
    const resultDiv = document.getElementById('result');

    // Listen for messages from the background script
    chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
        if (request.action === 'showResult') {
            const prediction = request.prediction;
            const probability = request.probability; // Update to use probability instead of confidence

            // Update the result div with prediction and probability
            resultDiv.innerHTML = `
                <p class="${prediction}">
                    Prediction: ${prediction.toUpperCase()}
                </p>
                <p>
                    Probability: ${(probability * 100).toFixed(2)}%
                </p>
            `;
            
            // Add color coding
            resultDiv.classList.remove('real', 'fake');
            resultDiv.classList.add(prediction);
        }
    });
});
