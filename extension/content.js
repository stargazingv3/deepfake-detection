document.body.addEventListener('click', function(event) {
    if (event.target.tagName === 'IMG') {
        const img = event.target;
        console.log('Image clicked:', img.src);  // Check if this fires correctly

        // Convert the image to base64
        const imgElement = img;
        const canvas = document.createElement("canvas");
        canvas.width = imgElement.naturalWidth;
        canvas.height = imgElement.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(imgElement, 0, 0);
        const dataURL = canvas.toDataURL("image/jpeg");

        // Send the base64 data URL to the background script
        chrome.runtime.sendMessage({ action: 'classifyImage', imageBase64: dataURL });
    }
});
