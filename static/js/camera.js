document.addEventListener('DOMContentLoaded', function() {
    const videoFeed = document.getElementById('videoFeed');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusDiv = document.getElementById('status');
    const errorDiv = document.getElementById('errorMessage');

    // Initialize video feed
    function initVideoFeed() {
        videoFeed.src = "/video_feed?" + Date.now();
        videoFeed.onerror = function() {
            errorDiv.textContent = "Video feed error. Retrying...";
            errorDiv.style.display = 'block';
            setTimeout(initVideoFeed, 2000);
        };
    }

    // Check detection status
    async function checkStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            statusDiv.textContent = `Status: ${data.running ? "Running" : "Not Running"}`;
            statusDiv.style.color = data.running ? "green" : "red";
            startBtn.disabled = data.running;
            stopBtn.disabled = !data.running;
        } catch (error) {
            console.error("Status check error:", error);
        }
    }

    // Event listeners
    startBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (data.status === "error") {
                throw new Error(data.message);
            }
            initVideoFeed();
            checkStatus();
        } catch (error) {
            errorDiv.textContent = error.message;
            errorDiv.style.display = 'block';
        }
    });

    stopBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            await response.json();
            checkStatus();
        } catch (error) {
            errorDiv.textContent = "Failed to stop detection";
            errorDiv.style.display = 'block';
        }
    });

    // Initialize
    initVideoFeed();
    checkStatus();
    setInterval(checkStatus, 3000);
});