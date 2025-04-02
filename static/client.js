// Advanced client for YOLO video processing with model selection and video file support

let ws = null;
let isConnected = false;
let isProcessing = false;
let videoStream = null;
let currentModel = 'yolov8n';
let modelType = 'standard'; // 'standard' or 'custom'
let customModelId = null;
let videoSource = 'video';  // Default to video file now
let videoIsPlaying = false;
let videoFile = null;
let frameQuality = 0.6;  // Reduced quality for faster processing
let processingInterval = 200; // Default processing interval in ms (adjust based on model)
let lastFrameProcessed = 0;
let frameSkipCount = 0;
let confidenceThreshold = 0.25;
let showConfidenceScores = true;
let vehicleTrackingEnabled = false;
let vehicleCounts = {
    left_to_right: {},
    right_to_left: {},
    total: {},
    total_left_to_right: 0,
    total_right_to_left: 0,
    all_total: 0
};

// DOM elements
const startButton = document.getElementById('startBtn');
const stopButton = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const modelSelect = document.getElementById('modelSelect');
const standardModelTab = document.getElementById('standardModelTab');
const customModelTab = document.getElementById('customModelTab');
const localVideo = document.getElementById('localVideo');
const localCanvas = document.getElementById('localCanvas');
const localCtx = localCanvas.getContext('2d');
const processedCanvas = document.getElementById('processedCanvas');
const processedCtx = processedCanvas.getContext('2d');
const fpsCounter = document.getElementById('fps');
const logContainer = document.getElementById('log');

// Video file elements
const cameraInputRadio = document.getElementById('cameraInput');
const videoInputRadio = document.getElementById('videoInput');
const videoFileInput = document.getElementById('videoFile');
const videoFileContainer = document.getElementById('videoFileContainer');
const videoControls = document.getElementById('videoControls');
const playPauseBtn = document.getElementById('playPauseBtn');
const videoSeek = document.getElementById('videoSeek');
const videoTime = document.getElementById('videoTime');

// Custom model elements
const customModelFile = document.getElementById('customModelFile');
const customModelFileName = document.getElementById('customModelFileName');
const customModelName = document.getElementById('customModelName');
const uploadModelBtn = document.getElementById('uploadModelBtn');
const uploadStatus = document.getElementById('uploadStatus');

// Performance controls
const qualitySlider = document.getElementById('qualitySlider') || { value: 60 };
const frameRateSlider = document.getElementById('frameRateSlider') || { value: 5 };
const resolutionSelect = document.getElementById('resolutionSelect') || { value: 'medium' };
const showConfidenceCheckbox = document.getElementById('showConfidence');
const confidenceThresholdSlider = document.getElementById('confidenceThreshold');

// Statistics display
const currentModelDisplay = document.getElementById('currentModel');
const processingTimeDisplay = document.getElementById('processingTime');
const frameSizeDisplay = document.getElementById('frameSize');
const detectionCountDisplay = document.getElementById('detectionCount');

// Vehicle tracking elements
const enableVehicleTrackingCheckbox = document.getElementById('enableVehicleTracking');
const trackingControls = document.getElementById('trackingControls');
const directionLinePosition = document.getElementById('directionLinePosition');
const linePositionValue = document.getElementById('linePositionValue');
const resetCountsBtn = document.getElementById('resetCountsBtn');
const vehicleCountDisplay = document.getElementById('vehicleCountDisplay');
const leftToRightCountDisplay = document.getElementById('leftToRightCount');
const rightToLeftCountDisplay = document.getElementById('rightToLeftCount');
const totalVehiclesDisplay = document.getElementById('totalVehicles');
const countTableBody = document.getElementById('countTableBody');

const vehicleCountingStats = document.getElementById('vehicleCountingStats');
const totalVehicleCount = document.getElementById('totalVehicleCount');
const leftToRightTotal = document.getElementById('leftToRightTotal');
const rightToLeftTotal = document.getElementById('rightToLeftTotal');
const countingTableBody = document.getElementById('countingTableBody');

// FPS tracking
let frameCount = 0;
let framesSent = 0;
let framesReceived = 0;
let lastFrameTime = 0;
let frameInterval = null;
let animationFrame = null;
let processingStartTime = 0;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Set up button listeners
    startButton.addEventListener('click', startStream);
    stopButton.addEventListener('click', stopStream);
    
    // Get available models
    fetchModels();
    
    // Set up model selection - adjust processing interval based on model
    modelSelect.addEventListener('change', function() {
        const previousModel = currentModel;
        currentModel = this.value;
        log(`Model changed to ${currentModel}`);
        
        // Update display
        currentModelDisplay.textContent = currentModel;
        
        // Adjust processing parameters based on model size
        adjustProcessingParams(currentModel);
        
        if (isConnected) {
            sendModelChange(currentModel);
        }
    });
    
    // Tab switching
    standardModelTab.addEventListener('click', function() {
        modelType = 'standard';
        // If connected, switch to selected standard model
        if (isConnected && modelSelect.value) {
            currentModel = modelSelect.value;
            sendModelChange(currentModel);
            currentModelDisplay.textContent = currentModel;
        }
    });
    
    customModelTab.addEventListener('click', function() {
        modelType = 'custom';
        // If connected and we have a custom model, switch to it
        if (isConnected && customModelId) {
            currentModel = customModelId;
            sendModelChange(currentModel);
            currentModelDisplay.textContent = `Custom: ${customModelName.value || 'Uploaded Model'}`;
        }
    });
    
    // Initialize performance controls if they exist
    if (qualitySlider) {
        qualitySlider.addEventListener('change', function() {
            frameQuality = this.value / 100;
            log(`Frame quality set to ${Math.round(frameQuality * 100)}%`);
        });
    }
    
    if (frameRateSlider) {
        frameRateSlider.addEventListener('change', function() {
            processingInterval = 1000 / this.value;
            log(`Target processing rate: ${this.value} FPS`);
        });
    }
    
    if (resolutionSelect) {
        resolutionSelect.addEventListener('change', function() {
            setResolution(this.value);
        });
    }
    
    if (showConfidenceCheckbox) {
        showConfidenceCheckbox.addEventListener('change', function() {
            showConfidenceScores = this.checked;
        });
    }
    
   if (confidenceThresholdSlider) {
    confidenceThresholdSlider.addEventListener('change', function() {
        confidenceThreshold = this.value / 100;
        log(`Confidence threshold set to ${confidenceThreshold.toFixed(2)}`);
        
        if (isConnected) {
            sendConfidenceChange(confidenceThreshold);
        }
    });
}
    
    // Set up input source selection
    cameraInputRadio.addEventListener('change', function() {
        if (this.checked) {
            videoSource = 'camera';
            videoFileContainer.classList.add('hidden');
            videoControls.style.display = 'none';
            log('Input source: Camera');
        }
    });
    
    videoInputRadio.addEventListener('change', function() {
        if (this.checked) {
            videoSource = 'video';
            videoFileContainer.classList.remove('hidden');
            log('Input source: Video File');
        }
    });
    
    videoFileInput.addEventListener('change', function(e) {
        if (e.target.files && e.target.files[0]) {
            videoFile = e.target.files[0];
            log(`Video file selected: ${videoFile.name}`);
        }
    });
    
    // Video controls
    playPauseBtn.addEventListener('click', function() {
        if (videoIsPlaying) {
            localVideo.pause();
            this.textContent = 'Play';
            videoIsPlaying = false;
        } else {
            localVideo.play();
            this.textContent = 'Pause';
            videoIsPlaying = true;
        }
    });
    
    videoSeek.addEventListener('input', function() {
        if (localVideo.duration) {
            const seekTime = (this.value / 100) * localVideo.duration;
            localVideo.currentTime = seekTime;
        }
    });
    
    localVideo.addEventListener('timeupdate', function() {
        if (localVideo.duration) {
            // Update seek bar
            videoSeek.value = (localVideo.currentTime / localVideo.duration) * 100;
            
            // Update time display
            const currentMinutes = Math.floor(localVideo.currentTime / 60);
            const currentSeconds = Math.floor(localVideo.currentTime % 60);
            const totalMinutes = Math.floor(localVideo.duration / 60);
            const totalSeconds = Math.floor(localVideo.duration % 60);
            
            videoTime.textContent = 
                `${currentMinutes.toString().padStart(2, '0')}:${currentSeconds.toString().padStart(2, '0')} / ` + 
                `${totalMinutes.toString().padStart(2, '0')}:${totalSeconds.toString().padStart(2, '0')}`;
        }
    });
    
    localVideo.addEventListener('play', function() {
        videoIsPlaying = true;
        playPauseBtn.textContent = 'Pause';
    });
    
    localVideo.addEventListener('pause', function() {
        videoIsPlaying = false;
        playPauseBtn.textContent = 'Play';
    });
    
    localVideo.addEventListener('ended', function() {
        videoIsPlaying = false;
        playPauseBtn.textContent = 'Play';
    });
    
    // Custom model upload and use immediately
    uploadModelBtn.addEventListener('click', uploadAndUseCustomModel);
    
    // Custom filename display
    customModelFile.addEventListener('change', function() {
        if (this.files.length > 0) {
            customModelFileName.textContent = this.files[0].name;
        } else {
            customModelFileName.textContent = 'No file chosen';
        }
    });
    
    // Vehicle tracking controls
    if (enableVehicleTrackingCheckbox) {
        enableVehicleTrackingCheckbox.addEventListener('change', function() {
            vehicleTrackingEnabled = this.checked;
			
			// Toggle the visibility of the counting statistics display
			if (vehicleCountingStats) {
				vehicleCountingStats.classList.toggle('hidden', !this.checked);
			}
            
            if (trackingControls) {
                trackingControls.classList.toggle('hidden', !this.checked);
            }
            
            if (vehicleCountDisplay) {
                vehicleCountDisplay.classList.toggle('hidden', !this.checked);
            }
            
            if (isConnected) {
                toggleVehicleTracking(this.checked);
            }
            
            log(`Vehicle tracking ${this.checked ? 'enabled' : 'disabled'}`);
        });
    }
    
    if (directionLinePosition) {
        directionLinePosition.addEventListener('input', function() {
            if (linePositionValue) {
                linePositionValue.textContent = this.value + '%';
            }
        });
        
        directionLinePosition.addEventListener('change', function() {
            if (isConnected && vehicleTrackingEnabled) {
                updateDirectionLine();
            }
        });
    }
    
    if (resetCountsBtn) {
        resetCountsBtn.addEventListener('click', function() {
            resetVehicleCounts();
        });
    }
    
    // Test connection
    testModels();
    
    // Update UI
    updateUI();
});

// Toggle vehicle tracking function
function toggleVehicleTracking(enabled) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log("Cannot toggle vehicle tracking - WebSocket not connected", true);
        return;
    }
    
    vehicleTrackingEnabled = enabled;
    log(`${enabled ? 'Enabling' : 'Disabling'} vehicle tracking`);
	console.log(`Vehicle tracking ${enabled ? 'enabled' : 'disabled'}, sending to server...`);
    
    ws.send(JSON.stringify({
        type: 'toggle_vehicle_tracking',
        enabled: enabled
    }));
	
	log(`${enabled ? 'Enabled' : 'Disabled'} vehicle tracking`);
    console.log("Current tracking state:", vehicleTrackingEnabled);
    
    // Reset counters in UI if disabled
    if (!enabled) {
        resetVehicleCounts();
    }
    
    // Show/hide the dedicated vehicle counting display
    if (vehicleCountingStats) {
        vehicleCountingStats.classList.toggle('hidden', !enabled);
		console.log(`Vehicle counting stats display is now ${!enabled ? 'hidden' : 'visible'}`);
    }
    
    // Also update the tracking controls visibility
    if (trackingControls) {
        trackingControls.classList.toggle('hidden', !enabled);
    }
}

function updateVehicleCountingDisplay() {
    // Update the total counts
    if (totalVehicleCount) {
        totalVehicleCount.textContent = vehicleCounts.all_total || 0;
    }
    
    if (leftToRightTotal) {
        leftToRightTotal.textContent = vehicleCounts.total_left_to_right || 0;
    }
    
    if (rightToLeftTotal) {
        rightToLeftTotal.textContent = vehicleCounts.total_right_to_left || 0;
    }
    
    // Update the detailed table
    if (countingTableBody) {
        // Clear the table
        countingTableBody.innerHTML = '';
        
        // Get all object classes from both directions
        const allClasses = new Set([
            ...Object.keys(vehicleCounts.left_to_right || {}),
            ...Object.keys(vehicleCounts.right_to_left || {})
        ]);
        
        // Sort classes by total count (highest first)
        const sortedClasses = Array.from(allClasses).sort((a, b) => 
            (vehicleCounts.total[b] || 0) - (vehicleCounts.total[a] || 0)
        );
        
        // Add rows for each object class
        sortedClasses.forEach(className => {
            const leftToRight = vehicleCounts.left_to_right[className] || 0;
            const rightToLeft = vehicleCounts.right_to_left[className] || 0;
            const total = vehicleCounts.total[className] || 0;
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${className}</td>
                <td>${leftToRight}</td>
                <td>${rightToLeft}</td>
                <td>${total}</td>
            `;
            countingTableBody.appendChild(row);
        });
        
        // If no data, add a "No data yet" row
        if (sortedClasses.length === 0) {
            const emptyRow = document.createElement('tr');
            emptyRow.innerHTML = `
                <td colspan="4" style="text-align: center; font-style: italic;">No vehicles counted yet</td>
            `;
            countingTableBody.appendChild(emptyRow);
        }
    }
}
// Reset vehicle counts
function resetVehicleCounts() {
    vehicleCounts = {
        left_to_right: {},
        right_to_left: {},
        total: {},
        total_left_to_right: 0,
        total_right_to_left: 0,
        all_total: 0
    };
    updateVehicleCountingDisplay();
    
    log('Vehicle counts reset');
    
    // Also notify server to reset counts if connected
    if (isConnected && vehicleTrackingEnabled) {
        ws.send(JSON.stringify({
            type: 'toggle_vehicle_tracking',
            enabled: true,  // This will reset counters on server
            reset: true
        }));
    }
}

// Update direction line position
function updateDirectionLine() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !vehicleTrackingEnabled) {
		console.log("Cannot update direction line - conditions not met");
        return;
    }
    
    // Convert percentage to absolute position
    const percentage = parseInt(directionLinePosition.value);
    
    // We'll send this to the server and let it calculate the actual pixel position
    // based on the current frame height
    log(`Updating direction line position to ${percentage}%`);
    
    ws.send(JSON.stringify({
        type: 'update_direction_line',
        percentage: percentage
    }));
	 log(`Sent direction line update: ${percentage}%`);
}

// Function to update the vehicle count display
function updateVehicleCountDisplay() {
    // Update summary counts
    if (leftToRightCountDisplay) {
        leftToRightCountDisplay.textContent = vehicleCounts.total_left_to_right;
    }
    
    if (rightToLeftCountDisplay) {
        rightToLeftCountDisplay.textContent = vehicleCounts.total_right_to_left;
    }
    
    if (totalVehiclesDisplay) {
        totalVehiclesDisplay.textContent = vehicleCounts.all_total;
    }
    
    // Update the detailed table
    if (countTableBody) {
        // Clear the table
        countTableBody.innerHTML = '';
        
        // Get all labels
        const allLabels = new Set([
            ...Object.keys(vehicleCounts.left_to_right),
            ...Object.keys(vehicleCounts.right_to_left)
        ]);
        
        // Sort labels by total count
        const sortedLabels = Array.from(allLabels).sort((a, b) => 
            (vehicleCounts.total[b] || 0) - (vehicleCounts.total[a] || 0)
        );
        
        // Add rows for each label
        sortedLabels.forEach(label => {
            const leftToRight = vehicleCounts.left_to_right[label] || 0;
            const rightToLeft = vehicleCounts.right_to_left[label] || 0;
            const total = vehicleCounts.total[label] || 0;
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${label}</td>
                <td>${leftToRight}</td>
                <td>${rightToLeft}</td>
                <td>${total}</td>
            `;
            countTableBody.appendChild(row);
        });
    }
}

// Upload and immediately use custom model
async function uploadAndUseCustomModel() {
    if (!customModelFile.files || !customModelFile.files[0]) {
        uploadStatus.textContent = 'Please select a model file (.pt)';
        uploadStatus.style.color = '#c62828';
        return;
    }
    
    // Check file extension
    const modelFile = customModelFile.files[0];
    if (!modelFile.name.toLowerCase().endsWith('.pt')) {
        uploadStatus.textContent = 'File must be a PyTorch model (.pt)';
        uploadStatus.style.color = '#c62828';
        return;
    }
    
    const name = customModelName.value || `custom_${Date.now()}`;
    
    uploadStatus.textContent = 'Uploading model...';
    uploadStatus.style.color = '#2196F3';
    
    const formData = new FormData();
    formData.append('model', modelFile);
    formData.append('name', name);
    
    try {
        log(`Uploading custom model: ${name} (${modelFile.name})`);
        
        const response = await fetch('/upload_model', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            uploadStatus.textContent = `Success: ${result.message}`;
            uploadStatus.style.color = '#4CAF50';
            log(`Custom model uploaded: ${name}, ID: ${result.model_id}`);
            
            customModelId = result.model_id;
            log(`Custom model ID: ${customModelId}`);
            
            // Set to custom model mode
            modelType = 'custom';
            customModelTab.click();
            
            // If connected, switch to the custom model
            if (isConnected) {
                currentModel = customModelId;
                sendModelChange(currentModel);
                currentModelDisplay.textContent = `Custom: ${name}`;
            }
        } else {
            uploadStatus.textContent = `Error: ${result.message}`;
            uploadStatus.style.color = '#c62828';
            log(`Model upload failed: ${result.message}`, true);
        }
    } catch (error) {
        uploadStatus.textContent = `Error: ${error.message}`;
        uploadStatus.style.color = '#c62828';
        log(`Model upload error: ${error.message}`, true);
    }
}

// Adjust processing parameters based on model
function adjustProcessingParams(modelName) {
    // Adjust quality and interval based on model size
    if (modelName.includes('n')) {  // nano
        processingInterval = 150;  // faster for small models
        frameQuality = 0.65;
    } else if (modelName.includes('s')) {  // small
        processingInterval = 200;
        frameQuality = 0.6;
    } else if (modelName.includes('m')) {  // medium
        processingInterval = 300;
        frameQuality = 0.55;
    } else if (modelName.includes('l')) {  // large
        processingInterval = 500;
        frameQuality = 0.5;
    } else if (modelName.includes('x')) {  // xlarge
        processingInterval = 800;
        frameQuality = 0.45;
    } else if (modelName.includes('custom')) {  // custom models
        processingInterval = 400;
        frameQuality = 0.5;
    }
    
    // Update UI sliders if they exist
    if (qualitySlider) qualitySlider.value = frameQuality * 100;
    if (frameRateSlider) frameRateSlider.value = Math.round(1000 / processingInterval);
    
    // Update displayed values
    if (document.getElementById('qualityValue')) {
        document.getElementById('qualityValue').textContent = Math.round(frameQuality * 100) + '%';
    }
    if (document.getElementById('frameRateValue')) {
        document.getElementById('frameRateValue').textContent = Math.round(1000 / processingInterval) + ' FPS';
    }
    
    log(`Adjusted processing: ${Math.round(frameQuality * 100)}% quality, ${Math.round(1000 / processingInterval)} FPS target`);
}

// Set video resolution
function setResolution(resolution) {
    let width, height;
    
    switch(resolution) {
        case 'low':
            width = 320;
            height = 240;
            break;
        case 'medium':
            width = 640;
            height = 480;
            break;
        case 'high':
            width = 1280;
            height = 720;
            break;
        default:
            width = 640;
            height = 480;
    }
    
    // Apply to canvas size - use smaller canvas for faster processing
    localCanvas.width = width;
    localCanvas.height = height;
    processedCanvas.width = width;
    processedCanvas.height = height;
    
    // Update frame size display
    frameSizeDisplay.textContent = `${width}×${height}`;
    
    log(`Resolution set to ${width}x${height}`);
}

// Update UI based on connection state
function updateUI() {
    startButton.disabled = isConnected;
    stopButton.disabled = !isConnected;
    modelSelect.disabled = isConnected;
    cameraInputRadio.disabled = isConnected;
    videoInputRadio.disabled = isConnected;
    videoFileInput.disabled = isConnected || videoSource !== 'video';
    qualitySlider.disabled = isConnected;
    frameRateSlider.disabled = isConnected;
    resolutionSelect.disabled = isConnected;
    uploadModelBtn.disabled = isConnected;
    customModelFile.disabled = isConnected;
    customModelName.disabled = isConnected;
    
    // Disable tab switching if connected
    standardModelTab.style.pointerEvents = isConnected ? 'none' : 'auto';
    customModelTab.style.pointerEvents = isConnected ? 'none' : 'auto';
    
    if (isConnected) {
        statusDiv.textContent = 'Connected';
        statusDiv.className = 'connected';
    } else {
        statusDiv.textContent = 'Disconnected';
        statusDiv.className = 'disconnected';
        
        // Reset statistics
        currentModelDisplay.textContent = '-';
        processingTimeDisplay.textContent = '-';
        frameSizeDisplay.textContent = '-';
        detectionCountDisplay.textContent = '-';
    }
}

// Fetch available models with better logging
async function fetchModels() {
    try {
        log('Fetching available models...');
        
        const response = await fetch('/models');
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }
        
        const models = await response.json();
        log(`Received ${Object.keys(models).length} models from server`);
        
        // Populate model select
        modelSelect.innerHTML = '';
        
        // First add standard models
        const standardModels = {
            'yolov8n': models['yolov8n'] || { description: 'YOLOv8 Nano - Fast and lightweight' },
            'yolov8s': models['yolov8s'] || { description: 'YOLOv8 Small - Good balance' },
            'yolov8m': models['yolov8m'] || { description: 'YOLOv8 Medium - Higher accuracy' },
            'yolov8l': models['yolov8l'] || { description: 'YOLOv8 Large - High accuracy' },
            'yolov8x': models['yolov8x'] || { description: 'YOLOv8 XLarge - Highest accuracy' }
        };
        
        // Create options for standard models
        Object.entries(standardModels).forEach(([id, info]) => {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = `${id} - ${info.description}`;
            modelSelect.appendChild(option);
        });
        
        // Check for custom models
        let foundCustomModels = false;
        Object.entries(models).forEach(([id, info]) => {
            if (id.startsWith('custom_')) {
                foundCustomModels = true;
                log(`Found custom model: ${id}, path: ${info.path}`);
                
                // If we don't have a custom model set, use this one
                if (!customModelId) {
                    customModelId = id;
                }
            }
        });
        
        // If we found custom models, enable the custom tab
        if (foundCustomModels) {
            customModelTab.style.opacity = 1;
        } else {
            customModelTab.style.opacity = 0.6;
        }
        
        log('Models loaded and dropdown updated');
    } catch (error) {
        log(`Error loading models: ${error.message}`, true);
    }
}

// Test model connectivity
async function testModels() {
    try {
        log('Testing model connectivity...');
        const response = await fetch('/test_models');
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            log(`Ultralytics version: ${data.ultralytics_version}`);
            log(`CUDA available: ${data.cuda_available ? 'Yes' : 'No'}`);
            
            Object.entries(data.model_tests).forEach(([model, status]) => {
                log(`Model ${model} test: ${status}`, status !== 'OK');
            });
        } else {
            log(`Model test failed: ${data.message}`, true);
        }
    } catch (error) {
        log(`Error testing models: ${error.message}`, true);
    }
}

// Start video stream
async function startStream() {
    try {
        // Determine which model to use based on mode
        if (modelType === 'custom') {
            if (customModelId) {
                currentModel = customModelId;
                currentModelDisplay.textContent = `Custom: ${customModelName.value || 'Uploaded Model'}`;
                log(`Using custom model: ${customModelId}`);
            } else {
                // No custom model available, fall back to standard
                modelType = 'standard';
                currentModel = modelSelect.value;
                standardModelTab.click();
                log(`No custom model available, using standard model: ${currentModel}`);
            }
        } else {
            // Standard model mode
            currentModel = modelSelect.value;
            log(`Using standard model: ${currentModel}`);
        }
        
        // Apply performance settings
        adjustProcessingParams(currentModel);
        
        // Connect WebSocket first
        await connectWebSocket();
        
        // Get video input based on selected source
        if (videoSource === 'camera') {
            await startCamera();
        } else {
            await loadVideoFile();
        }
        
        // Reset counters
        frameCount = 0;
        framesSent = 0;
        framesReceived = 0;
        lastFrameTime = performance.now();
        lastFrameProcessed = 0;
        frameSkipCount = 0;
        
        // Start frame processing
        startFrameProcessing();
        
        isConnected = true;
        updateUI();
        
        // Enable vehicle tracking if checkbox is checked
        if (enableVehicleTrackingCheckbox && enableVehicleTrackingCheckbox.checked) {
            toggleVehicleTracking(true);
        } else {
    // Ensure the counting display is hidden
    if (vehicleCountingStats) {
        vehicleCountingStats.classList.add('hidden');
    }
}
        
        log(`Processing started with ${modelType} model: ${currentModel}`);
    } catch (error) {
        log(`Error starting stream: ${error.message}`, true);
        
        // Clean up if there was an error
        if (ws) {
            ws.close();
            ws = null;
        }
        
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
        }
    }
}

// Stop video stream
function stopStream() {
    // Stop frame processing
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
    
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }
    
    // Stop video stream (for camera)
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    
    // Reset video player
    localVideo.pause();
    localVideo.removeAttribute('src');
    localVideo.load();
    videoControls.style.display = 'none';
    videoIsPlaying = false;
    
    // Close WebSocket
    if (ws) {
        ws.close();
        ws = null;
    }
    
    // Clear canvases
    localCtx.clearRect(0, 0, localCanvas.width, localCanvas.height);
    processedCtx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
    
    isConnected = false;
    isProcessing = false;
    updateUI();
    log('Stream stopped');
}

// Connect to WebSocket server
async function connectWebSocket() {
    return new Promise((resolve, reject) => {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
            const wsUrl = `${protocol}${window.location.host}/ws`;
            
            log(`Connecting to WebSocket: ${wsUrl}`);
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                log('WebSocket connected');
                
                // Send initial model selection and confidence threshold
                sendModelChange(currentModel);
                sendConfidenceChange(confidenceThreshold);
                
                resolve();
            };
            
            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    
                    if (message.type === 'processed_frame') {
                        // Track processing time
                        const processingTime = performance.now() - processingStartTime;
                        processingTimeDisplay.textContent = `${processingTime.toFixed(0)}ms`;
                        
                        // Update detections count
                        if (message.detections !== undefined) {
                            detectionCountDisplay.textContent = message.detections;
                        }
                        
                        // Process vehicle counts if available
                        if (message.vehicle_counts) {
                            // Update vehicle counts
							console.log("Received vehicle counts:", message.vehicle_counts);
                            vehicleCounts = message.vehicle_counts;
                            
                            // Update the display
                            if (vehicleTrackingEnabled) {
								updateVehicleCountingDisplay();
								updateVehicleCountDisplay();
								console.log("Updated vehicle count displays with:", vehicleCounts);
                            }
                        }
                        
                        // Display the processed frame
                        displayProcessedFrame(message.data);
                        
                        // Ready for next frame
                        isProcessing = false;
                        framesReceived++;
                    } else if (message.type === 'status') {
                        log(`Server: ${message.message}`);
                    } else if (message.type === 'error') {
                        log(`Server error: ${message.message}`, true);
                        isProcessing = false;  // Don't block on errors
                    } else if (message.type === 'pong') {
                        // Ping-pong for connection testing
                        const roundTripTime = Date.now() - message.timestamp;
                        log(`WebSocket ping: ${roundTripTime}ms`);
                    }
                } catch (error) {
                    log(`Error parsing WebSocket message: ${error.message}`, true);
                    isProcessing = false;  // Don't block on errors
                }
            };
            
            ws.onclose = (event) => {
                log(`WebSocket disconnected: code=${event.code}`, true);
                isConnected = false;
                updateUI();
            };
            
            ws.onerror = (error) => {
                const errorMsg = 'WebSocket error';
                log(errorMsg, true);
                reject(new Error(errorMsg));
            };
            
            // Add timeout
            setTimeout(() => {
                if (ws && ws.readyState !== WebSocket.OPEN) {
                    reject(new Error('WebSocket connection timeout'));
                }
            }, 5000);
        } catch (error) {
            reject(error);
        }
    });
}

// Send confidence threshold change to server
function sendConfidenceChange(confidence) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log("Cannot send confidence change - WebSocket not connected", true);
        return;
    }
    
    log(`Sending confidence threshold change: ${confidence.toFixed(2)}`);
    
    ws.send(JSON.stringify({
        type: 'confidence_change',
        value: confidence
    }));
}

// Start camera
async function startCamera() {
    try {
        log('Requesting camera access...');
        
        // Get resolution constraints
        const width = parseInt(localCanvas.width) || 640;
        const height = parseInt(localCanvas.height) || 480;
        
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: width },
                height: { ideal: height }
            }
        });
        
        // Display video in element
        localVideo.srcObject = videoStream;
        await localVideo.play();
        
        // Adjust canvas size based on actual video dimensions
        const videoTrack = videoStream.getVideoTracks()[0];
        const settings = videoTrack.getSettings();
        
        localCanvas.width = settings.width || width;
        localCanvas.height = settings.height || height;
        processedCanvas.width = settings.width || width;
        processedCanvas.height = settings.height || height;
        
        // Update frame size display
        frameSizeDisplay.textContent = `${settings.width || width}×${settings.height || height}`;
        
        log(`Camera started: ${settings.width}x${settings.height}`);
    } catch (error) {
        log('Camera access denied or not available', true);
        throw error;
    }
}

// Load video file
async function loadVideoFile() {
    return new Promise((resolve, reject) => {
        if (!videoFile) {
            reject(new Error('No video file selected'));
            return;
        }
        
        log(`Loading video file: ${videoFile.name}`);
        
        const videoURL = URL.createObjectURL(videoFile);
        localVideo.src = videoURL;
        
        // Show video controls
        videoControls.style.display = 'flex';
        
        // Set up events
        localVideo.onloadedmetadata = () => {
            log(`Video loaded: ${Math.round(localVideo.duration)}s, ${localVideo.videoWidth}x${localVideo.videoHeight}`);
            
            // Set canvas dimensions to match video but respect resolution setting
            const maxWidth = parseInt(localCanvas.width) || 640;
            const aspectRatio = localVideo.videoHeight / localVideo.videoWidth;
            
            let width = Math.min(localVideo.videoWidth, maxWidth);
            let height = Math.round(width * aspectRatio);
            
            localCanvas.width = width;
            localCanvas.height = height;
            processedCanvas.width = width;
            processedCanvas.height = height;
            
            // Update frame size display
            frameSizeDisplay.textContent = `${width}×${height}`;
            
            // Reset video to beginning
            localVideo.currentTime = 0;
            
            // Start playing
            localVideo.play()
                .then(() => {
                    videoIsPlaying = true;
                    playPauseBtn.textContent = 'Pause';
                    resolve();
                })
                .catch(error => {
                    log(`Error playing video: ${error.message}`, true);
                    reject(error);
                });
        };
        
        localVideo.onerror = (error) => {
            log(`Error loading video: ${error.message || 'Unknown error'}`, true);
            reject(new Error('Video loading failed'));
        };
    });
}

// Send model change to server with better logging
function sendModelChange(modelName) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log("Cannot send model change - WebSocket not connected", true);
        return;
    }
    
    log(`Sending model change to server: ${modelName}`);
    
    ws.send(JSON.stringify({
        type: 'model_change',
        model: modelName
    }));
    
    // Update display
    currentModelDisplay.textContent = modelType === 'custom' && customModelId === modelName ? 
        `Custom: ${customModelName.value || 'Uploaded Model'}` : modelName;
}

// Start processing video frames
function startFrameProcessing() {
    // Use requestAnimationFrame for more efficient processing
    const processFrameLoop = () => {
        const now = performance.now();
        const elapsed = now - lastFrameProcessed;
        
        // Check if it's time to process a new frame based on the interval
        if (!isProcessing && elapsed >= processingInterval) {
            captureAndSendFrame();
            lastFrameProcessed = now;
        }
        
        // Update FPS counter every second
        if (now - lastFrameTime > 1000) {
            const processingTime = now - lastFrameTime;
            const fps = Math.round((framesSent * 1000) / processingTime);
            const receivedFps = Math.round((framesReceived * 1000) / processingTime);
            
            fpsCounter.textContent = `${receivedFps} FPS (Send: ${fps})`;
            
            framesSent = 0;
            framesReceived = 0;
            lastFrameTime = now;
            
            // Log periodic statistics
            if (frameSkipCount > 0) {
                log(`Performance: ${frameSkipCount} frames skipped due to backpressure`);
                frameSkipCount = 0;
            }
        }
        
        // Continue loop if connected
        if (isConnected) {
            animationFrame = requestAnimationFrame(processFrameLoop);
        }
    };
    
    // Start the processing loop
    animationFrame = requestAnimationFrame(processFrameLoop);
}
// Capture and send a frame to the server
function captureAndSendFrame() {
    if (!isConnected || !ws || ws.readyState !== WebSocket.OPEN) {
        return;
    }
    
    // Skip frame if we're still processing previous one
    if (isProcessing) {
        frameSkipCount++;
        return;
    }
    
    try {
        // Draw video to canvas (potentially at reduced size)
        localCtx.drawImage(localVideo, 0, 0, localCanvas.width, localCanvas.height);
        
        // Convert to base64 with quality setting
        const frameData = localCanvas.toDataURL('image/jpeg', frameQuality);
        
        // Record start time for this frame
        processingStartTime = performance.now();
        
        // Send to server
        ws.send(JSON.stringify({
            type: 'frame',
            data: frameData
        }));
        
        isProcessing = true;
        framesSent++;
        frameCount++;
    } catch (error) {
        log(`Error capturing frame: ${error.message}`, true);
        isProcessing = false; // Reset processing flag on error
    }
}

// Display processed frame
function displayProcessedFrame(data) {
    try {
        const img = new Image();
        img.onload = () => {
            processedCtx.drawImage(img, 0, 0, processedCanvas.width, processedCanvas.height);
        };
        img.onerror = (error) => {
            log(`Error loading processed frame: ${error}`, true);
        };
        img.src = data;
    } catch (error) {
        log(`Error displaying processed frame: ${error.message}`, true);
    }
}

// Add log entry
function log(message, isError = false) {
    const entry = document.createElement('div');
    entry.className = isError ? 'log-error' : 'log-info';
    
    const time = new Date().toTimeString().split(' ')[0];
    entry.textContent = `[${time}] ${message}`;
    
    logContainer.insertBefore(entry, logContainer.firstChild);
    
    // Limit log entries
    if (logContainer.children.length > 100) {
        logContainer.removeChild(logContainer.lastChild);
    }
    
    console.log(`${isError ? '[ERROR]' : '[INFO]'} ${message}`);
}

// Send ping to test WebSocket connection
function pingWebSocket() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log('WebSocket not connected', true);
        return;
    }
    
    ws.send(JSON.stringify({
        type: 'ping',
        timestamp: Date.now()
    }));
}