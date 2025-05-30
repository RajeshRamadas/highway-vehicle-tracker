import os
import asyncio
import json
import cv2
import numpy as np
import base64
import logging
import aiohttp
from aiohttp import web, WSMsgType
from ultralytics import YOLO
import torch
from concurrent.futures import ThreadPoolExecutor
import uuid
import shutil
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize thread pool for processing
thread_pool = ThreadPoolExecutor(max_workers=2)

# Directory for custom models
CUSTOM_MODEL_DIR = "custom_models"
os.makedirs(CUSTOM_MODEL_DIR, exist_ok=True)

# Create models directory for downloaded models
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# YOLO model configurations - Updated with more YOLOv8 variants
YOLO_VERSIONS = {
    'yolov8n': {
        'path': os.path.join(MODELS_DIR, 'yolov8n.pt'),
        'description': 'YOLOv8 Nano - Fast and lightweight',
        'type': 'detect'
    },
    'yolov8s': {
        'path': os.path.join(MODELS_DIR, 'yolov8s.pt'),
        'description': 'YOLOv8 Small - Good balance of speed and accuracy',
        'type': 'detect'
    },
    'yolov8m': {
        'path': os.path.join(MODELS_DIR, 'yolov8m.pt'),
        'description': 'YOLOv8 Medium - Higher accuracy, moderate speed',
        'type': 'detect'
    },
    'yolov8l': {
        'path': os.path.join(MODELS_DIR, 'yolov8l.pt'),
        'description': 'YOLOv8 Large - High accuracy, slower',
        'type': 'detect'
    },
    'yolov8x': {
        'path': os.path.join(MODELS_DIR, 'yolov8x.pt'),
        'description': 'YOLOv8 XLarge - Highest accuracy, slowest',
        'type': 'detect'
    }
}

# Configure S3 client
def configure_s3_client():
    # Load environment variables for S3 if not already loaded
    if 'AWS_ACCESS_KEY_ID' not in os.environ:
        load_dotenv()
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        region_name=os.getenv('AWS_REGION'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    return s3_client

# Active websocket connections
websockets = {}

# Connection data
connection_data = {}

# Function to process frame with YOLO
async def process_frame(frame_data, model_version, connection_id, confidence=0.25):
    try:
        logger.info(f"Processing frame with model: {model_version}")
        
        # Decode base64 image
        img_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            return {'image': frame_data, 'error': 'Failed to decode image'}
        
        # Initialize connection data if not exists
        if connection_id not in connection_data:
            connection_data[connection_id] = {
                'models': {},
                'vehicle_tracking': {
                    'enabled': False,
                    'left_to_right_counts': defaultdict(int),
                    'right_to_left_counts': defaultdict(int),
                    'total_counts': defaultdict(int),
                    'tracked_vehicles': {},
                    'next_vehicle_id': 0,
                    'direction_line_y': None,
                    'direction_line_threshold': 10,
                    'frame_count': 0
                },
                'last_frame_height': None
            }
        
        # Store frame dimensions for percentage calculations
        height, width = img.shape[:2]
        connection_data[connection_id]['last_frame_height'] = height
        
        # Get vehicle tracking settings
        vehicle_tracking = connection_data[connection_id]['vehicle_tracking']
        track_vehicles = vehicle_tracking['enabled']
        
        # Initialize direction line if not set
        if track_vehicles and vehicle_tracking['direction_line_y'] is None:
            vehicle_tracking['direction_line_y'] = height // 2
            logger.info(f"Initialized vehicle tracking line at y={vehicle_tracking['direction_line_y']}")
        
        # Check if there's a pending line percentage to apply
        if track_vehicles and 'pending_line_percentage' in connection_data[connection_id]:
            percentage = connection_data[connection_id]['pending_line_percentage']
            vehicle_tracking['direction_line_y'] = int((percentage / 100) * height)
            logger.info(f"Applied pending direction line at {percentage}% (y={vehicle_tracking['direction_line_y']})")
            del connection_data[connection_id]['pending_line_percentage']
        
        # Get model configuration - ensure proper handling for custom models
        if model_version in YOLO_VERSIONS:
            model_config = YOLO_VERSIONS[model_version]
            logger.info(f"Found model config for {model_version}: {model_config}")
        else:
            logger.error(f"Model version {model_version} not found in YOLO_VERSIONS")
            return {
                'image': frame_data, 
                'error': f'Model {model_version} not found',
                'detections': 0
            }
            
        model_type = model_config.get('type', 'detect')
        
        # Use thread pool to not block the event loop
        def run_inference():
            # Initialize or get model
            models = connection_data[connection_id]['models']
            
            if model_version not in models:
                try:
                    model_path = model_config['path']
                    logger.info(f"Loading model from path: {model_path}")
                    
                    # Check if the model file exists
                    if not os.path.exists(model_path):
                        logger.error(f"Model file not found at path: {model_path}")
                        raise FileNotFoundError(f"Model file not found: {model_path}")
                    
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    logger.info(f"Using device: {device}")
                    model = YOLO(model_path)
                    model.to(device)
                    models[model_version] = model
                    logger.info(f"Model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    raise
            
            model = models[model_version]
            
            # Run inference
            try:
                results = model(img, verbose=False, conf=confidence)
                logger.info(f"Inference completed with results: {bool(results)}")
                return results[0] if results else None
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                raise
        
        # Run inference in thread pool
        result = await asyncio.get_event_loop().run_in_executor(thread_pool, run_inference)
        
        # Update frame count for tracking
        if track_vehicles:
            vehicle_tracking['frame_count'] += 1
            logger.info(f"Processing frame {vehicle_tracking['frame_count']} with tracking enabled, line at y={vehicle_tracking['direction_line_y']}")
        
        detection_count = 0
        vehicle_count_data = None
        
        if result:
            # Draw on image based on model type
            if model_type == 'detect':
                # Draw a counting line if vehicle tracking is enabled
                if track_vehicles:
                    direction_line_y = vehicle_tracking['direction_line_y']
                    # Make line more visible - thicker and brighter blue
                    cv2.line(img, (0, direction_line_y), (width, direction_line_y), (255, 0, 0), 1)
                    
                    # Add text to indicate line purpose - larger font
                    #cv2.putText(img, "COUNTING LINE", (width//4, direction_line_y - 15),
                    #          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
                
                # Process detections
                for box in result.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = result.names[cls_id]
                    
                    # Draw rectangle and text
                    color = (0, 255, 0)  # Green for regular detections
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    detection_count += 1
                    
                    # Vehicle tracking logic
                    if track_vehicles:
                        # Center point of the detected object
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Simple tracking based on overlap with previous detections
                        vehicle_matched = False
                        frame_count = vehicle_tracking['frame_count']
                        tracked_vehicles = vehicle_tracking['tracked_vehicles']
                        direction_line_y = vehicle_tracking['direction_line_y']
                        direction_line_threshold = vehicle_tracking['direction_line_threshold']
                        
                        # Check if this detection matches any tracked vehicle
                        for vehicle_id, vehicle_data in list(tracked_vehicles.items()):
                            prev_x1, prev_y1, prev_x2, prev_y2 = vehicle_data['coords']
                            
                            # Check for overlap
                            if (x1 < prev_x2 and x2 > prev_x1 and 
                                y1 < prev_y2 and y2 > prev_y1):
                                
                                # Update tracked vehicle
                                tracked_vehicles[vehicle_id] = {
                                    'coords': (x1, y1, x2, y2),
                                    'center_y': center_y,
                                    'label': cls_name,
                                    'crossed_line': vehicle_data.get('crossed_line', False),
                                    'direction': vehicle_data.get('direction', None),
                                    'last_seen': frame_count
                                }
                                
                                # Check if vehicle crossed the counting line
                                prev_center_y = (prev_y1 + prev_y2) // 2
                                
                                # Determine direction and count if line is crossed
                                if not vehicle_data.get('crossed_line', False):
                                    vehicle_label = vehicle_data.get('label', cls_name)
                                    
                                    if (prev_center_y < direction_line_y and 
                                        center_y >= direction_line_y - direction_line_threshold):
                                        # Vehicle moving from top to bottom (left to right)
                                        vehicle_tracking['left_to_right_counts'][vehicle_label] += 1
                                        vehicle_tracking['total_counts'][vehicle_label] += 1
                                        tracked_vehicles[vehicle_id]['crossed_line'] = True
                                        tracked_vehicles[vehicle_id]['direction'] = 'left_to_right'
                                        logger.info(f"DETECTED: Vehicle {vehicle_id} ({vehicle_label}) crossed line LEFT TO RIGHT")
                                        
                                    elif (prev_center_y > direction_line_y and 
                                          center_y <= direction_line_y + direction_line_threshold):
                                        # Vehicle moving from bottom to top (right to left)
                                        vehicle_tracking['right_to_left_counts'][vehicle_label] += 1
                                        vehicle_tracking['total_counts'][vehicle_label] += 1
                                        tracked_vehicles[vehicle_id]['crossed_line'] = True
                                        tracked_vehicles[vehicle_id]['direction'] = 'right_to_left'
                                        logger.info(f"DETECTED: Vehicle {vehicle_id} ({vehicle_label}) crossed line RIGHT TO LEFT")
                                
                                vehicle_matched = True
                                break
                        
                        # If no match found, add as new vehicle
                        if not vehicle_matched:
                            tracked_vehicles[vehicle_tracking['next_vehicle_id']] = {
                                'coords': (x1, y1, x2, y2),
                                'center_y': center_y,
                                'label': cls_name,
                                'crossed_line': False,
                                'direction': None,
                                'last_seen': frame_count
                            }
                            vehicle_tracking['next_vehicle_id'] += 1
                
                # Remove vehicles that haven't been seen for a while (5 frames)
                if track_vehicles:
                    frame_count = vehicle_tracking['frame_count']
                    tracked_vehicles = vehicle_tracking['tracked_vehicles']
                    for vehicle_id in list(tracked_vehicles.keys()):
                        if frame_count - tracked_vehicles[vehicle_id]['last_seen'] > 5:
                            del tracked_vehicles[vehicle_id]
            
            # Add counting statistics to the response if tracking is enabled
            if track_vehicles:
                # Prepare the counting data to send to the client
                vehicle_count_data = {
                    'left_to_right': dict(vehicle_tracking['left_to_right_counts']),
                    'right_to_left': dict(vehicle_tracking['right_to_left_counts']),
                    'total': dict(vehicle_tracking['total_counts']),
                    'total_left_to_right': sum(vehicle_tracking['left_to_right_counts'].values()),
                    'total_right_to_left': sum(vehicle_tracking['right_to_left_counts'].values()),
                    'all_total': sum(vehicle_tracking['total_counts'].values())
                }
            
            # Add processing info on the image
            #cv2.putText(img, f"Model: {model_version}", (10, 30),
            #          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Encode the processed image to base64
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            logger.info(f"Frame processed successfully with {detection_count} detections")
            response = {
                'image': f"data:image/jpeg;base64,{img_str}",
                'status': 'success',
                'detections': detection_count
            }
            
            # Add vehicle counting data if available
            if vehicle_count_data:
                response['vehicle_counts'] = vehicle_count_data
                
            return response
        else:
            logger.warning("No results from model inference")
            return {'image': frame_data, 'status': 'no_results', 'detections': 0}
    
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return {'image': frame_data, 'error': str(e), 'detections': 0}

# WebSocket handler
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # Generate a unique ID for this connection
    ws_id = id(ws)
    websockets[ws_id] = {'ws': ws, 'model_version': 'yolov8n', 'confidence': 0.25}
    logger.info(f"New WebSocket connection: {ws_id}")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    # Handle different message types
                    if data['type'] == 'model_change':
                        # Update the model selection
                        model_version = data['model']
                        if model_version in YOLO_VERSIONS:
                            websockets[ws_id]['model_version'] = model_version
                            await ws.send_json({
                                'type': 'status',
                                'message': f"Model changed to {model_version}"
                            })
                            logger.info(f"Model changed to {model_version} for connection {ws_id}")
                        else:
                            logger.error(f"Invalid model requested: {model_version}")
                            await ws.send_json({
                                'type': 'error',
                                'message': f"Invalid model: {model_version}"
                            })
                    
                    elif data['type'] == 'frame':
                        # Process a video frame
                        model_version = websockets[ws_id]['model_version']
                        confidence = websockets[ws_id].get('confidence', 0.25)
                        logger.info(f"Received frame from client, processing with {model_version} (conf: {confidence})")
                        
                        processed_data = await process_frame(data['data'], model_version, ws_id, confidence)
                        
                        # Send back the processed frame
                        if 'error' in processed_data:
                            await ws.send_json({
                                'type': 'error',
                                'message': processed_data['error']
                            })
                        else:
                            await ws.send_json({
                                'type': 'processed_frame',
                                'data': processed_data['image'],
                                'status': processed_data.get('status', 'success'),
                                'detections': processed_data.get('detections', 0),
                                'vehicle_counts': processed_data.get('vehicle_counts', None)
                            })
                            logger.info("Sent processed frame back to client")
                    
                    elif data['type'] == 'confidence_change':
                        # Update confidence threshold
                        confidence = float(data['value'])
                        if 0 <= confidence <= 1:
                            websockets[ws_id]['confidence'] = confidence
                            await ws.send_json({
                                'type': 'status',
                                'message': f"Confidence threshold set to {confidence:.2f}"
                            })
                            logger.info(f"Confidence threshold updated to {confidence:.2f}")
                        else:
                            await ws.send_json({
                                'type': 'error',
                                'message': f"Invalid confidence value: {confidence}"
                            })
                    
                    elif data['type'] == 'toggle_vehicle_tracking':
                        # Toggle vehicle tracking
                        tracking_enabled = data.get('enabled', False)
                        if ws_id in connection_data:
                            connection_data[ws_id]['vehicle_tracking']['enabled'] = tracking_enabled
                            
                            # Reset counters if turning on tracking or explicitly requested
                            if tracking_enabled or data.get('reset', False):
                                connection_data[ws_id]['vehicle_tracking'] = {
                                    'enabled': True,
                                    'left_to_right_counts': defaultdict(int),
                                    'right_to_left_counts': defaultdict(int),
                                    'total_counts': defaultdict(int),
                                    'tracked_vehicles': {},
                                    'next_vehicle_id': 0,
                                    'direction_line_y': None,
                                    'direction_line_threshold': 10,
                                    'frame_count': 0
                                }
                            
                            await ws.send_json({
                                'type': 'status',
                                'message': f"Vehicle tracking {'enabled' if tracking_enabled else 'disabled'}"
                            })
                            logger.info(f"Vehicle tracking {'enabled' if tracking_enabled else 'disabled'} for connection {ws_id}")
                    
                    elif data['type'] == 'update_direction_line':
                        # Update the direction line position
                        percentage = data.get('percentage', None)
                        if ws_id in connection_data and percentage is not None:
                            # Calculate actual pixel position based on latest frame height
                            if 'last_frame_height' in connection_data[ws_id]:
                                height = connection_data[ws_id]['last_frame_height']
                                y_position = int((percentage / 100) * height)
                                connection_data[ws_id]['vehicle_tracking']['direction_line_y'] = y_position
                                await ws.send_json({
                                    'type': 'status',
                                    'message': f"Direction line updated to {percentage}% (y={y_position})"
                                })
                                logger.info(f"Direction line updated to {percentage}% (y={y_position}) for connection {ws_id}")
                            else:
                                # If we don't have frame height yet, store percentage and apply later
                                connection_data[ws_id]['pending_line_percentage'] = percentage
                                await ws.send_json({
                                    'type': 'status',
                                    'message': f"Direction line will be set to {percentage}% on next frame"
                                })
                    
                    elif data['type'] == 'ping':
                        # Simple ping to check connection
                        await ws.send_json({
                            'type': 'pong',
                            'timestamp': data.get('timestamp', 0)
                        })
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    await ws.send_json({
                        'type': 'error',
                        'message': "Invalid JSON format"
                    })
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await ws.send_json({
                        'type': 'error',
                        'message': f"Server error: {str(e)}"
                    })
            
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket connection closed with exception: {ws.exception()}")
    
    finally:
        if ws_id in websockets:
            del websockets[ws_id]
        if ws_id in connection_data:
            del connection_data[ws_id]
        logger.info(f"WebSocket connection closed: {ws_id}")
    
    return ws

# Set up template files
def setup_templates():
    """Create template files for S3 browser and model download pages"""
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create s3_browser.html if it doesn't exist
    if not os.path.exists('templates/s3_browser.html'):
        try:
            with open('s3_browser.html', 'r') as template:
                with open('templates/s3_browser.html', 'w') as f:
                    f.write(template.read())
            logger.info("Created S3 browser template")
        except FileNotFoundError:
            logger.warning("s3_browser.html template not found, please create it manually")
    """
    # Create download_models.html if it doesn't exist
    if not os.path.exists('templates/download_models.html'):
        try:
            with open('download_models.html', 'r') as template:
                with open('templates/download_models.html', 'w') as f:
                    f.write(template.read())
            logger.info("Created model download template")
        except FileNotFoundError:
            logger.warning("download_models.html template not found, please create it manually")
    """
async def on_prepare(request, response):
    # CORS headers here if needed
    pass
    
# Get available models endpoint
async def get_models(request):
    return web.json_response(YOLO_VERSIONS)

# Test connection endpoint
async def test_models(request):
    try:
        # Test if YOLO can be imported
        import ultralytics
        yolo_version = ultralytics.__version__
        
        # Test if models can be loaded
        test_results = {}
        for model_name, model_info in YOLO_VERSIONS.items():
            try:
                model = YOLO(model_info['path'])
                test_results[model_name] = "OK"
            except Exception as e:
                test_results[model_name] = f"Error: {str(e)}"
        
        return web.json_response({
            'status': 'success',
            'ultralytics_version': yolo_version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'model_tests': test_results
        })
    except Exception as e:
        return web.json_response({
            'status': 'error',
            'message': str(e)
        }, status=500)

# Handle custom model upload
async def upload_custom_model(request):
    try:
        data = await request.post()
        
        if 'model' not in data or not data['model'].file:
            logger.error("No model file in upload request")
            return web.json_response({
                'status': 'error',
                'message': 'No model file uploaded'
            }, status=400)
        
        model_file = data['model'].file
        model_name = data.get('name', '')
        
        if not model_name:
            model_name = f"custom_model_{uuid.uuid4().hex[:8]}"
        
        # Make the model ID more reliable by removing special characters
        model_id = f"custom_{model_name.lower().replace(' ', '_').replace('-', '_')}"
        # Further clean the ID to remove any special chars
        model_id = ''.join(c for c in model_id if c.isalnum() or c == '_')
        
        logger.info(f"Processing model upload: name='{model_name}', id='{model_id}'")
        
        # Ensure the model ID is unique
        if model_id in YOLO_VERSIONS:
            logger.warning(f"Model ID '{model_id}' already exists in YOLO_VERSIONS")
            return web.json_response({
                'status': 'error',
                'message': f'Model with name {model_name} already exists'
            }, status=400)
        
        # Save the uploaded file
        os.makedirs(CUSTOM_MODEL_DIR, exist_ok=True)
        file_path = os.path.join(CUSTOM_MODEL_DIR, f"{model_id}.pt")
        
        with open(file_path, 'wb') as f:
            model_file.seek(0)
            shutil.copyfileobj(model_file, f)
        
        logger.info(f"Custom model saved to {file_path}")
        
        # Validate the model
        try:
            logger.info(f"Validating model at {file_path}")
            model = YOLO(file_path)
            model_type = 'detect'  # Default to detect
            
            # Add to models list
            YOLO_VERSIONS[model_id] = {
                'path': file_path,
                'description': f'Custom model: {model_name}',
                'type': model_type
            }
            
            logger.info(f"Added custom model to YOLO_VERSIONS with id={model_id}, path={file_path}")
            
            return web.json_response({
                'status': 'success',
                'message': f'Model {model_name} uploaded successfully',
                'model_id': model_id
            })
        except Exception as e:
            # If validation fails, delete the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logger.error(f"Error validating uploaded model: {str(e)}")
            return web.json_response({
                'status': 'error',
                'message': f'Invalid model file: {str(e)}'
            }, status=400)
    
    except Exception as e:
        logger.error(f"Error handling model upload: {str(e)}")
        return web.json_response({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }, status=500)

# S3 Browser Functions
async def s3_browser(request):
    """Render the S3 browser page"""
    return web.FileResponse('templates/s3_browser.html')

async def list_s3_files(request):
    """API endpoint to list files in the S3 bucket"""
    try:
        s3_client = configure_s3_client()
        BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
        
        prefix = request.query.get('prefix', '')
        delimiter = '/'
        
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=prefix,
            Delimiter=delimiter
        )
        
        # Process folders (common prefixes)
        folders = []
        if 'CommonPrefixes' in response:
            for prefix_item in response['CommonPrefixes']:
                prefix_path = prefix_item['Prefix']
                # Extract just the folder name, not the full path
                folder_name = prefix_path
                if prefix:
                    if prefix_path.startswith(prefix):
                        folder_name = prefix_path[len(prefix):]
                
                # Remove trailing slash for display
                if folder_name.endswith('/'):
                    folder_name = folder_name[:-1]
                
                folders.append({
                    'key': prefix_path,
                    'name': folder_name,
                    'is_folder': True,
                    'size': 0,
                    'last_modified': None
                })
        
        # Process files
        files = []
        if 'Contents' in response:
            for item in response['Contents']:
                # Skip the current directory entry itself or folders
                if item['Key'] == prefix or item['Key'].endswith('/'):
                    continue
                
                # Extract just the file name, not the full path
                file_name = item['Key']
                if prefix:
                    if file_name.startswith(prefix):
                        file_name = file_name[len(prefix):]
                
                files.append({
                    'key': item['Key'],
                    'name': file_name,
                    'is_folder': False,
                    'size': item['Size'],
                    'last_modified': item['LastModified'].isoformat()
                })
        
        # Create breadcrumb data
        breadcrumbs = []
        if prefix:
            # Add root
            breadcrumbs.append({
                'name': 'Root',
                'path': '',
                'is_root': True
            })
            
            # Split the prefix by '/' to build breadcrumb trail
            parts = prefix.split('/')
            current_path = ''
            
            # Last item might be empty if prefix ends with '/'
            if parts[-1] == '':
                parts.pop()
            
            for i, part in enumerate(parts):
                current_path += part + '/'
                breadcrumbs.append({
                    'name': part,
                    'path': current_path,
                    'is_root': False
                })
        else:
            # Just the root breadcrumb
            breadcrumbs.append({
                'name': 'Root',
                'path': '',
                'is_root': True
            })
        
        return web.json_response({
            'files': files,
            'folders': folders,
            'breadcrumbs': breadcrumbs,
            'current_prefix': prefix
        })
        
    except ClientError as e:
        logger.error(f"Error listing S3 files: {e}")
        return web.json_response({'error': str(e)}, status=500)
    except Exception as e:
        logger.error(f"Unexpected error listing S3 files: {e}")
        return web.json_response({'error': 'An unexpected error occurred'}, status=500)

async def download_s3_file(request):
    """Generate a presigned URL for file download"""
    try:
        s3_client = configure_s3_client()
        BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
        
        file_key = request.query.get('key')
        if not file_key:
            return web.json_response({'error': 'File key is required'}, status=400)
            
        # Generate a presigned URL for direct download
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': file_key
            },
            ExpiresIn=900  # 15 minutes
        )
        
        return web.json_response({'download_url': presigned_url})
        
    except ClientError as e:
        logger.error(f"Error generating download URL: {e}")
        return web.json_response({'error': str(e)}, status=500)
    except Exception as e:
        logger.error(f"Unexpected error generating download URL: {e}")
        return web.json_response({'error': 'An unexpected error occurred'}, status=500)


async def download_model_page(request):
    """Render the model download page"""
    return web.FileResponse('templates/download_models.html')

async def list_yolo_models(request):
    """API endpoint to list available YOLO models for download"""
    try:
        # List of available YOLO models for download
        models = [
            {
                'id': 'yolov8n',
                'name': 'YOLOv8 Nano',
                'description': 'Fast and lightweight model for real-time detection',
                'size': '6.3 MB',
                'download_url': '/api/download-model?model=yolov8n.pt'
            },
            {
                'id': 'yolov8s',
                'name': 'YOLOv8 Small',
                'description': 'Good balance between speed and accuracy',
                'size': '21.5 MB',
                'download_url': '/api/download-model?model=yolov8s.pt'
            },
            {
                'id': 'yolov8m',
                'name': 'YOLOv8 Medium',
                'description': 'Higher accuracy, moderate speed',
                'size': '52.0 MB',
                'download_url': '/api/download-model?model=yolov8m.pt'
            },
            {
                'id': 'yolov8l',
                'name': 'YOLOv8 Large',
                'description': 'High accuracy, slower speed',
                'size': '86.5 MB',
                'download_url': '/api/download-model?model=yolov8l.pt'
            },
            {
                'id': 'yolov8x',
                'name': 'YOLOv8 XLarge',
                'description': 'Highest accuracy, slowest speed',
                'size': '130.5 MB',
                'download_url': '/api/download-model?model=yolov8x.pt'
            }
        ]
        
        return web.json_response(models)
    except Exception as e:
        logger.error(f"Error listing YOLO models: {e}")
        return web.json_response({'error': 'An unexpected error occurred'}, status=500)

async def download_model(request):
    """API endpoint to download a specific YOLO model"""
    try:
        s3_client = configure_s3_client()
        BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
        
        model_name = request.query.get('model')
        if not model_name:
            return web.json_response({'error': 'Model name is required'}, status=400)
        
        # First check if model exists locally
        model_id = model_name.split('.')[0]  # Remove extension
        
        local_path = None
        # Check in YOLO_VERSIONS for standard models
        if model_id in YOLO_VERSIONS:
            local_path = YOLO_VERSIONS[model_id]['path']
        # Check custom models directory
        elif model_id.startswith('custom_'):
            local_path = os.path.join(CUSTOM_MODEL_DIR, model_name)
        
        # If model exists locally, serve it directly
        if local_path and os.path.exists(local_path):
            logger.info(f"Serving local model: {local_path}")
            return web.FileResponse(
                path=local_path,
                headers={'Content-Disposition': f'attachment; filename="{model_name}"'}
            )
        
        # If not found locally, try S3
        try:
            logger.info(f"Checking S3 for model: {model_name}")
            # Generate a presigned URL for the model in S3
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': BUCKET_NAME,
                    'Key': f'models/{model_name}'
                },
                ExpiresIn=900  # 15 minutes
            )
            logger.info(f"Model found in S3, returning presigned URL")
            return web.json_response({'download_url': presigned_url})
        except ClientError as e:
            logger.warning(f"Model not found in S3: {e}")
            
            # If not in S3, download from Ultralytics and serve
            # Only for standard YOLOv8 models
            if model_id in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']:
                try:
                    # Create models directory if it doesn't exist
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    target_path = os.path.join(MODELS_DIR, model_name)
                    
                    # Check if we already downloaded it
                    if not os.path.exists(target_path):
                        logger.info(f"Downloading model from Ultralytics: {model_name}")
                        
                        # Download model using aiohttp
                        ultraytics_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
                        
                        async with aiohttp.ClientSession() as session:
                            async with session.get(ultraytics_url) as response:
                                if response.status != 200:
                                    raise Exception(f"Failed to download model: HTTP {response.status}")
                                
                                # Save model to file
                                with open(target_path, 'wb') as f:
                                    f.write(await response.read())
                        
                        logger.info(f"Model downloaded and saved to {target_path}")
                    else:
                        logger.info(f"Using previously downloaded model: {target_path}")
                    
                    # Serve the downloaded file
                    return web.FileResponse(
                        path=target_path,
                        headers={'Content-Disposition': f'attachment; filename="{model_name}"'}
                    )
                
                except Exception as e:
                    logger.error(f"Error downloading model from Ultralytics: {e}")
                    # Fall back to redirect to Ultralytics
                    return web.json_response({
                        'error': 'Could not download model',
                        'fallback_url': f'https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}'
                    }, status=404)
            else:
                # For custom models that are not found
                return web.json_response({
                    'error': 'Model not found',
                    'message': f'The model {model_name} was not found in local storage or S3'
                }, status=404)
                
    except Exception as e:
        logger.error(f"Error downloading YOLO model: {e}")
        return web.json_response({'error': 'An unexpected error occurred'}, status=500)

# Main application setup
app = web.Application(
    client_max_size=1024*1024*1000,  # Set max size to 1000 MB (adjust as needed)
)
app.on_response_prepare.append(on_prepare)
app.router.add_get('/', lambda r: web.FileResponse('index.html'))
app.router.add_get('/ws', websocket_handler)
app.router.add_get('/models', get_models)
app.router.add_get('/test_models', test_models)
app.router.add_post('/upload_model', upload_custom_model)
app.router.add_static('/static/', path='static')

# Add S3 and model download routes
def add_s3_routes():
    """Add S3 browser and model download routes to the app"""
    # Check and create template files
    setup_templates()
    
    # Add routes
    app.router.add_get('/s3-browser', s3_browser)
    app.router.add_get('/api/s3/files', list_s3_files)
    app.router.add_get('/api/s3/download', download_s3_file)
    app.router.add_get('/download-model', download_model_page)
    app.router.add_get('/api/yolo-models', list_yolo_models)
    #app.router.add_get('/api/download-model', download_model)
    
    logger.info("S3 browser and model download routes added")

# Check if models exist
def check_models():
    for model_name, model_info in YOLO_VERSIONS.items():
        model_path = model_info['path']
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            
            # For standard models, suggest downloading
            if not model_path.startswith(CUSTOM_MODEL_DIR) and not model_path.startswith('models/'):
                logger.info(f"You can download this model using: pip install -U ultralytics && yolo download model={model_name}")

if __name__ == '__main__':
    logger.info("Starting server and checking models...")
    check_models()
    add_s3_routes()
    web.run_app(app, host='0.0.0.0', port=8081)