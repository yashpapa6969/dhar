let displayDiv = document.getElementById('textDisplay');
let serverAvailable = false;
let micAvailable = false;
let fullSentences = [];
const serverCheckInterval = 5000; // Check every 5 seconds

let socket; // Handle to the WebSocket

function setupWebSocket() {
    socket = new WebSocket("ws://97.119.112.191:40001");

    socket.onopen = () => {
        console.log("Connected to WebSocket server.");
        serverAvailable = true;
        updateStatusMessage();
    };

    socket.onmessage = (event) => {
        let data = JSON.parse(event.data);

        switch (data.type) {
      
            case 'final':
                fullSentences.push(data.text);
                displayRealtimeText(""); // Refresh display with new full sentence
                break;
        }
    };

    socket.onclose = () => {
        console.log("WebSocket connection closed.");
        serverAvailable = false;
        updateStatusMessage();
    };

    socket.onerror = (error) => {
        console.error("WebSocket error:", error);
        alert("WebSocket error - check console for details.");
    };
}

function displayRealtimeText(realtimeText) {
    let displayedText = fullSentences.map((sentence, index) => {
        let span = document.createElement('span');
        span.textContent = sentence + " ";
        span.className = index % 2 === 0 ? 'yellow' : 'cyan';
        return span.outerHTML;
    }).join('') + realtimeText;

    displayDiv.innerHTML = displayedText;
}

function updateStatusMessage() {
    if (!micAvailable)
        displayRealtimeText("🎤  please allow microphone access  🎤");
    else if (!serverAvailable)
        displayRealtimeText("🖥️  please start server  🖥️");
    else
        displayRealtimeText("👄  start speaking  👄");
}

// Initialize WebSocket connection and UI
setupWebSocket();
updateStatusMessage();


// Request access to the microphone
navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    let audioContext = new AudioContext();
    let source = audioContext.createMediaStreamSource(stream);
    let processor = audioContext.createScriptProcessor(256, 1, 1);

    source.connect(processor);
    processor.connect(audioContext.destination);
    micAvailable = true;
    updateStatusMessage();

    processor.onaudioprocess = function(e) {
        let inputData = e.inputBuffer.getChannelData(0);
        let outputData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
            outputData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
        }
        if (socket && socket.readyState === WebSocket.OPEN) {
            let metadata = JSON.stringify({ sampleRate: audioContext.sampleRate });
            let metadataBytes = new TextEncoder().encode(metadata);
            let metadataLength = new ArrayBuffer(4);
            new DataView(metadataLength).setInt32(0, metadataBytes.byteLength, true);
            let combinedData = new Blob([metadataLength, metadataBytes, outputData.buffer]);
            socket.send(combinedData);
        }
    };
}).catch(e => {
    console.error("Error accessing microphone:", e);
    alert("Error accessing microphone - check console for details.");
});
