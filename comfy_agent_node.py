# Filename: comfy_agent_node.py
# Place this file in your ComfyUI/custom_nodes/ directory,
# or in a subdirectory like ComfyUI/custom_nodes/my_utility_nodes/
# If in a subdirectory, ensure you have an __init__.py file in that subdirectory
# that exports the NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.

import io
import os
import shutil
import uuid
import threading
import time
import requests
import json
import server # ComfyUI's server instance
import nodes
import subprocess
import logging
import traceback
import base64
import datetime
import urllib.parse

from server import PromptServer
from folder_paths import base_path, get_filename_list, get_user_directory, get_input_directory, get_directory_by_type, models_dir, folder_names_and_paths, recursive_search

from .dtos import (
    ComfyAgentConfig, RegisterComfyAgent, GetComfyAgentEvents, UpdateComfyAgent, UpdateComfyAgentStatus, UpdateWorkflowGeneration, GpuInfo, 
    CaptionArtifact, CompleteOllamaGenerateTask, GetOllamaGenerateTask, ComfyAgentSettings
)
from servicestack import JsonServiceClient, UploadFile, WebServiceException, ResponseStatus, EmptyResponse, printdump, from_json

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from .classifier import load_image_models, classify_image
from .audio_classifer import load_audio_model, get_audio_tags
from .imagehash import phash, dominant_color_hex

VERSION = 1
DEFAULT_AUTOSTART = True
DEFAULT_INSTALL_MODELS   = True
DEFAULT_INSTALL_NODES    = True
DEFAULT_INSTALL_PACKAGES = True
DEFAULT_ENDPOINT_URL = "https://ubixar.com"
DEFAULT_POLL_INTERVAL_SECONDS = 10
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60
DEVICE_ID = None
MIN_DOWNLOAD_BYTES=1024

# Stores the active configuration for the global poller
g_config = {
    'enabled':          DEFAULT_AUTOSTART,
    'install_models':   DEFAULT_INSTALL_MODELS,
    'install_nodes':    DEFAULT_INSTALL_NODES,
    'install_packages': DEFAULT_INSTALL_PACKAGES,
}
g_client = None
g_models = None
g_audio_model = None
g_headers_json={"Content-Type": "application/json"}
g_headers={}
g_needs_update = False
g_settings=ComfyAgentSettings(preserve_outputs=True)

g_logger_prefix = "[comfy-agent]"
g_node_dir = os.path.join(get_user_directory(), "comfy_agent")
g_running = False

g_categories = []
g_uploaded_prompts = []
g_language_models = None

# Install
g_statuses = []
g_installed_pip_packages = []
g_installed_custom_nodes = []
g_installed_models = []
g_downloading_model = None
g_downloading_model_args = None

MSG_RESTART_COMFY = "Restart ComfyUI for changes to take effect."
# --- End of Default Configuration ---

# Maintain a global dictionary of prompt_id mapping to client_id
g_pending_prompts = {}

def create_client():
    client = JsonServiceClient(config_str('url'))
    client.bearer_token = config_str('apikey')
    return client

def _log(message):
    """Helper method for logging from the global polling task."""
    print(f"{g_logger_prefix} {message}")

def _log_error(message, e):
    """Helper method for logging errors from the global polling task."""
    status = None
    if isinstance(e, requests.exceptions.HTTPError):
        try:
            dto = from_json(EmptyResponse, e.response.text)
            status = dto.response_status
        except:
            status = ResponseStatus(error_code='HTTPError', message=e.response.text)
    if isinstance(e, WebServiceException):
        status = e.response_status

    if status is not None:
        error_code = f"[{status.error_code}] " if status.error_code != 'Exception' else ""
        print(f"{g_logger_prefix} {message}{error_code}{status.message}")
    else:
        print(f"{g_logger_prefix} {message}{type(e)} {e}")

def is_enabled():
    global g_config
    return 'enabled' in g_config and g_config['enabled'] or False

def allow_installing_models():
    global g_config
    return 'install_models' in g_config and g_config['install_models'] or False

def allow_installing_nodes():
    global g_config
    return 'install_nodes' in g_config and g_config['install_nodes'] or False

def allow_installing_packages():
    global g_config
    return 'install_packages' in g_config and g_config['install_packages'] or False

def config_str(name:str):
    return name in g_config and g_config[name] or ""

def get_comfyui_version():
    # comfyui_version.py is a generated file that's sometimes not available
    if os.path.exists(os.path.join(base_path, "comfyui_version.py")):
        from comfyui_version import __version__
        return __version__
    return "unknown"

# Store the original method
original_send_sync = PromptServer.send_sync

# Define your interceptor function
def intercepted_send_sync(self, event, data, sid=None):
    # Your custom code to run before the event is sent
    if not event == "progress":
        _log(f"event={event}")

    if event == "executed" or event == "execution_success" or event == "status": 
        _log(json.dumps(data))
        # Do something with the execution data

    # Call the original method
    result = original_send_sync(self, event, data, sid)

    # Your custom code to run after the event is sent
    if event == "execution_success":
        _log("After execution_success event sent")
        prompt_id = data['prompt_id']
        if prompt_id in g_pending_prompts:
            client_id = g_pending_prompts[prompt_id]
            # call send_execution_success in a background thread
            threading.Thread(target=send_execution_success, args=(prompt_id, client_id), daemon=True).start()
    elif event == "execution_error":
        prompt_id = data['prompt_id']
        if prompt_id in g_pending_prompts:
            client_id = g_pending_prompts[prompt_id]
            _log("After execution_error event sent " + prompt_id)
            _log(json.dumps(data))
            exception_type = data['exception_type']
            exception_message = data['exception_message']
            traceback = data['traceback']
            # call send_execution_error in a background thread
            threading.Thread(target=send_execution_error,
                args=(prompt_id, client_id, exception_type, exception_message, traceback), daemon=True).start()

    return result

def remove_pending_prompt(prompt_id):
    if prompt_id in g_pending_prompts:
        del g_pending_prompts[prompt_id]

def send_execution_error(prompt_id, client_id, exception_type, exception_message, traceback):
    _log(f"send_execution_error: prompt_id={prompt_id}, client_id={client_id}")
    try:
        # only join first 5 lines of traceback
        stack_trace = "\n".join(traceback[:5])
        # split '.' and take last part
        message = f"{exception_type.split('.')[-1]}: {exception_message}"
        request = UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, prompt_id=prompt_id,
            queue_count=get_queue_count(),
            error=ResponseStatus(error_code=exception_type, message=message, stack_trace=stack_trace))
        g_client.post(request)
    except WebServiceException as ex:
        _log(f"Exception sending execution_error: {ex}")
        printdump(ex.response_status)
    except Exception as e:
        _log(f"Error sending execution_error: {e}")
    finally:
        remove_pending_prompt(prompt_id)

image_extensions = ["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"]
audio_extensions = ["mp3", "aac", "flac", "wav", "wma", "m4a", "ogg", "opus", "aiff"]

def send_execution_success(prompt_id, client_id):
    _log(f"send_execution_success: prompt_id={prompt_id}, client_id={client_id}")

    if prompt_id in g_uploaded_prompts:
        _log(f"prompt_id={prompt_id} already sent, skipping.")
        return

    try:
        result = PromptServer.instance.prompt_queue.get_history(prompt_id=prompt_id)
        if prompt_id not in result:
            _log(f"prompt_id={prompt_id} not found in history, skipping.")
            return
        prompt_data = result[prompt_id]
        outputs = prompt_data['outputs']
        status = prompt_data['status']
        _log(json.dumps(outputs))
        _log(json.dumps(status))

        # example image outputs:
        # {"10": {"images": [{"filename": "ComfyUI_temp_pgpib_00001_.png", "subfolder": "", "type": "temp"}]}}
        # example audio outputs:
        # {"13":{"audio":[{"filename":"ComfyUI_00001_.flac","subfolder":"audio","type":"output"}]}}

        #extract all image outputs
        artifacts = []
        for key, value in outputs.items():
            if 'images' in value:
                artifacts.extend(value['images'])
            if 'audio' in value:
                artifacts.extend(value['audio'])
        # outputs = {"images": artifacts}
        _log(json.dumps(artifacts, indent=2))

        files = []
        output_paths = []
        for artifact in artifacts:
            dir = get_directory_by_type(artifact['type'])
            artifact_path = os.path.join(dir, artifact['subfolder'], artifact['filename'])

            if not os.path.exists(artifact_path):
                _log(f"File not found: {artifact_path}")
                continue

            if artifact_path not in output_paths:
                output_paths.append(artifact_path)

            #lowercase extension
            ext = artifact['filename'].split('.')[-1].lower()

            if (ext in image_extensions):
                with Image.open(artifact_path) as img:
                    artifact['width'] = img.width
                    artifact['height'] = img.height
                    # convert png to webp
                    if ext == "png":
                        quality = 90
                        buffer = io.BytesIO()
                        img.save(buffer, format='webp', quality=quality)
                        buffer.seek(0)
                        image_stream = buffer
                        artifact['filename'] = artifact['filename'].replace(".png", ".webp")
                        ext = "webp"
                    else:
                        image_stream=open(artifact_path, 'rb')

                    metadata = classify_image(g_models, g_categories, img, debug=True)
                    artifact.update(metadata)
                    artifact['phash'] = f"{phash(img)}"
                    artifact['color'] = dominant_color_hex(img)

                files.append(UploadFile(
                    field_name=f"output_{len(files)}",
                    file_name=artifact['filename'],
                    content_type=f"image/{ext}",
                    stream=image_stream
                ))
            elif (ext in audio_extensions):
                if ext != "m4a":
                    to_aac_path = artifact_path.replace(f".{ext}", ".m4a")
                    bitrate = "192k"
                    artifact['codec'] = "aac"
                    try:
                        command = [
                            "ffmpeg",
                            "-i", artifact_path,
                            "-c:a", artifact['codec'], # Specify AAC audio codec
                            "-b:a", bitrate,           # Set audio bitrate
                            to_aac_path
                        ]
                        subprocess.run(command, check=True, capture_output=True, text=True)
                        artifact['filename'] = os.path.basename(to_aac_path)
                        _log(f"Audio conversion successful: {os.path.basename(artifact_path)} -> {artifact['filename']}")
                        artifact_path = to_aac_path
                        if artifact_path not in output_paths:
                            output_paths.append(artifact_path)
                        ext = "m4a"

                        command = [
                            'ffprobe',
                            '-v', 'quiet',  # Suppress verbose output
                            '-print_format', 'json',  # Output in JSON format
                            '-show_format',  # Show format information
                            #'-show_streams',  # Show stream information
                            artifact_path
                        ]
                        output_bytes = subprocess.check_output(command)
                        metadata_str = output_bytes.decode('utf-8')
                        metadata = json.loads(metadata_str)
                        _log("ffprobe metadata: ")
                        print(json.dumps(metadata, indent=2))

                        if 'format' in metadata:
                            format = metadata['format']
                            if 'size' in format:
                                artifact['length'] = int(format['size'])
                            if 'duration' in format:
                                artifact['duration'] = float(format['duration'])
                            if 'bit_rate' in format:
                                artifact['bitrate'] = int(format['bit_rate'])
                            if 'nb_streams' in format:
                                artifact['streams'] = int(format['nb_streams'])
                            if 'nb_programs' in format:
                                artifact['programs'] = int(format['nb_programs'])

                    except subprocess.CalledProcessError as e:
                        _log(f"Error during conversion: {e}")
                        print(f"FFmpeg output: {e.stdout}")
                        print(f"FFmpeg error: {e.stderr}")
                        continue
                    except FileNotFoundError:
                        _log("Error: FFmpeg not found. Please ensure it's installed and in your PATH.")
                        continue

                    global g_audio_model
                    if g_audio_model is None:
                        try:
                            g_audio_model = load_audio_model(models_dir=models_dir)
                        except Exception as ex:
                            _log(f"Error loading audio model: {ex}")

                    if g_audio_model is not None:
                        try:
                            tags = get_audio_tags(g_audio_model, artifact_path, debug=True)
                            artifact['tags'] = tags
                        except Exception as ex:
                            _log(f"Error getting audio tags: {ex}")

                files.append(UploadFile(
                    field_name=f"output_{len(files)}",
                    file_name=artifact['filename'],
                    content_type="audio/mp4",
                    stream=open(artifact_path, 'rb')
                ))
            else:
                _log(f"Unsupported file type: {ext}")

        request = UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, prompt_id=prompt_id,
            queue_count=get_queue_count(),
            outputs=json.dumps(outputs),
            status=json.dumps(status))
        g_client.post_files_with_request(request, files)
        g_uploaded_prompts.append(prompt_id)

        if not g_settings.preserve_outputs:
            for path in output_paths:
                try:
                    _log(f"Deleting output: {path}")
                    os.remove(path)
                except Exception as e:
                    _log(f"Error deleting file: {e}")

    except WebServiceException as ex:
        _log(f"Exception sending execution_success: {ex}")
        printdump(ex.response_status)
    except Exception as e:
        _log_error("Error sending execution_success: ", e)
        stack_trace = traceback.format_exc()
        logging.error(stack_trace)
        exception_type = type(e).__name__ or "Exception"
        send_execution_error(prompt_id, client_id, exception_type, str(e), stack_trace)
    finally:
        remove_pending_prompt(prompt_id)

def urljoin(*args):
    trailing_slash = '/' if args[-1].endswith('/') else ''
    return "/".join([str(x).strip("/") for x in args]) + trailing_slash

# copied from PromptServer
def node_info(node_class):
    obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
    info = {}
    info['input'] = obj_class.INPUT_TYPES()
    info['input_order'] = {key: list(value.keys()) for (key, value) in obj_class.INPUT_TYPES().items()}
    info['output'] = obj_class.RETURN_TYPES
    info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [False] * len(obj_class.RETURN_TYPES)
    info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
    info['name'] = node_class
    info['display_name'] = nodes.NODE_DISPLAY_NAME_MAPPINGS[node_class] if node_class in nodes.NODE_DISPLAY_NAME_MAPPINGS.keys() else node_class
    info['description'] = obj_class.DESCRIPTION if hasattr(obj_class,'DESCRIPTION') else ''
    info['python_module'] = getattr(obj_class, "RELATIVE_PYTHON_MODULE", "nodes")
    info['category'] = 'sd'
    if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE is True:
        info['output_node'] = True
    else:
        info['output_node'] = False

    if hasattr(obj_class, 'CATEGORY'):
        info['category'] = obj_class.CATEGORY

    if hasattr(obj_class, 'OUTPUT_TOOLTIPS'):
        info['output_tooltips'] = obj_class.OUTPUT_TOOLTIPS

    if getattr(obj_class, "DEPRECATED", False):
        info['deprecated'] = True
    if getattr(obj_class, "EXPERIMENTAL", False):
        info['experimental'] = True

    if hasattr(obj_class, 'API_NODE'):
        info['api_node'] = obj_class.API_NODE
    return info

def get_object_info():
    out = {}
    keys = list(nodes.NODE_CLASS_MAPPINGS.keys())
    for x in keys:
        try:
            out[x] = node_info(x)
        except Exception:
            logging.error(f"[ERROR] An error occurred while retrieving information for the '{x}' node.")
            logging.error(traceback.format_exc())
    return out

def get_object_info_json():
    return json.dumps(get_object_info())

def get_object_info_json_from_url():
    json = requests.get(f"{get_server_url()}/api/object_info").text
    return json

def listen_to_messages_poll():
    global g_client, g_running, g_needs_update
    g_running = True
    g_client = create_client()
    retry_secs = 5
    time.sleep(1)

    try:
        register_agent()
    except Exception as ex:
        _log_error("Error registering agent: ", ex)
        logging.error(traceback.format_exc())
        g_running = False
        return

    global g_models
    if g_models is None:
        try:
            g_models = load_image_models(models_dir=models_dir, debug=True)
        except Exception as ex:
            _log(f"Error loading image models: {ex}")
            g_running = False
            return

    while is_enabled():
        try:
            g_running = True
            send_update()

            if g_needs_update:
                g_needs_update = False
                g_client = create_client()
                register_agent()

            _log("Polling for agent events")
            request = GetComfyAgentEvents(device_id=DEVICE_ID)

            response = g_client.get(request)
            retry_secs = 5
            if response.results is not None:
                event_names = [event.name for event in response.results]
                _log(f"Processing {len(response.results)} agent events: {','.join(event_names)}")
                for event in response.results:
                    if event.name == "Register":
                        register_agent()
                    elif event.name == "ExecWorkflow":
                        inputs = 'inputs' in event.args and event.args['inputs'].split(',') or None
                        exec_prompt(event.args['url'], inputs)
                    elif event.name == "ExecOllama":
                        exec_ollama(event.args['model'], event.args['endpoint'], event.args['request'], event.args['replyTo'])
                    elif event.name == "CaptionImage":
                        caption_image(event.args['url'], event.args['model'])
                    elif event.name == "InstallPipPackage":
                        install_pip_package(event.args['package'])
                    elif event.name == "UninstallPipPackage":
                        uninstall_pip_package(event.args['package'])
                    elif event.name == "InstallCustomNode":
                        install_custom_node(event.args['url'])
                    elif event.name == "UninstallCustomNode":
                        uninstall_custom_node(event.args['url'])
                    elif event.name == "DownloadModel":
                        install_model(event.args['model'])
                    elif event.name == "DeleteModel":
                        delete_model(event.args['path'])
                    elif event.name == "Refresh":
                        time_str=datetime.datetime.now().strftime("%H:%M:%S")
                        send_update(status=f"Updated at {time_str}")
                    elif event.name == "Reboot":
                        reboot()
        except Exception as ex:
            _log(f"Error connecting to {config_str('url')}: {ex}, retrying in {retry_secs}s")
            time.sleep(retry_secs)  # Wait before retrying
            retry_secs += 5 # Exponential backoff
            g_client = create_client() # Create new client to force reconnect
    _log(f"Disconnected from {config_str('url')}")
    g_running = False

def get_queue_count():
    return PromptServer.instance.get_queue_info()['exec_info']['queue_remaining']

def resolve_url(url):
    #if relative path, combine with BASE_URL
    if not url.startswith("http"):
        url = urljoin(config_str('url'), url)
    return url

def get_server_url():
    return f"http://{PromptServer.instance.address}:{PromptServer.instance.port}"

def exec_prompt(url, inputs=None):

    if url is None:
        _log("exec_prompt: url is None")
        return

    # Get the server address - typically localhost when running within ComfyUI
    # server_address = PromptServer.instance.server_address
    # host, port = server_address if server_address else ("127.0.0.1", 7860)

    url = resolve_url(url)
    _log(f"exec_prompt GET: {url}")

    api_response = requests.get(url, headers=g_headers_json, timeout=30)
    if api_response.status_code != 200:
        _log(f"Error: {api_response.status_code} - {api_response.text}")
        return

    input_dir = get_input_directory()
    if inputs is not None:
        for input in inputs:
            input_url = resolve_url(input)
            filename = os.path.basename(input_url)
            input_file = os.path.join(input_dir, filename)
            if os.path.exists(input_file):
                _log(f"exec_prompt: input file already exists: {filename}")
                continue
            _log(f"exec_prompt GET: {input_url}")
            input_response = requests.get(input_url, timeout=30)
            if input_response.status_code != 200:
                _log(f"Error: {input_response.status_code} - {input_response.text}")
                return
            # save file to input folder
            with open(input_file, 'wb') as f:
                f.write(input_response.content)
            _log(f"exec_prompt saved input file: {filename}")

    prompt_data = api_response.json()
    if 'client_id' not in prompt_data:
        _log("Error: No client_id in prompt data")
        return

    client_id = prompt_data['client_id']

    # check if client_id is a value in g_pending_prompts
    for key, value in g_pending_prompts.items():
        if value == client_id:
            prompt_id = key
            _log(f"exec_prompt: client_id={client_id} already in progress prompt_id={prompt_id}")
            g_client.post(UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, prompt_id=prompt_id,
                queue_count=get_queue_count()))
            return

    _log(f"exec_prompt: /prompt client_id={client_id}")

    # Call the /prompt endpoint
    response = requests.post(
        f"{get_server_url()}/prompt",
        json=prompt_data,
        headers=g_headers_json)

    if response.status_code == 200:
        result = response.json()
        prompt_id = result['prompt_id']
        _log(f"exec_prompt: /prompt OK prompt_id={prompt_id}, client_id={client_id}")
        _log(json.dumps(result))

        g_pending_prompts[prompt_id] = client_id
        g_client.post(UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, prompt_id=prompt_id,
            queue_count=get_queue_count()))
    else:
        error_message = f"Error: {response.status_code} - {response.text}"
        _log(error_message)
        _log(json.dumps(prompt_data))
        g_client.post(UpdateWorkflowGeneration(device_id=DEVICE_ID, id=client_id, queue_count=get_queue_count(),
            error={"error_code": response.status_code, "message": response.text}))

def url_to_image(url):
    """Download an image from URL and return as PIL Image object"""
    try:
        response = requests.get(url) # Send GET request to download the image
        response.raise_for_status()  # Raises an HTTPError for bad responses
        image = Image.open(io.BytesIO(response.content)) # Create PIL Image from the downloaded bytes
        return image
    except requests.exceptions.RequestException as e:
        _log(f"Error downloading image: {e}")
        return None
    except Exception as e:
        _log_error("Error opening image: ", e)
        return None

def url_to_bytes(url):
    """Download an image from URL and return as PIL Image object"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        _log(f"Error downloading image: {e}")
        return None
    except Exception as e:
        _log_error("Error opening image: ", e)
        return None

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except FileNotFoundError:
        _log(f"Error: Image file '{image_path}' not found.")
        return None
    except Exception as e:
        _log_error("Error encoding image: ", e)
        return None

def exec_ollama(model:str, endpoint:str, request:str, reply_to):
    error = None

    try:
        if g_language_models is None:
            error = ResponseStatus(error_code='Validation', message="Ollama is not available")
        elif model is None:
            error = ResponseStatus(error_code='Validation', message="model is None")
        elif endpoint is None:
            error = ResponseStatus(error_code='Validation', message="endpoint is None")
        elif request is None:
            error = ResponseStatus(error_code='Validation', message="request is None")
        elif reply_to is None:
            error = ResponseStatus(error_code='Validation', message="replyTo is None")
        elif model not in g_language_models:
            error = ResponseStatus(error_code='Validation', message=f"model {model} is not available")

        reply_url = resolve_url(reply_to)

        if error is not None:
            if reply_to is None:
                _log(f"exec_ollama: {error.error_code} {error.message}")
            else:
                body = {
                    'response_status': error
                }
                g_client.post_url(reply_url, body)
            return

        try:
            ollama_request = request
            if ollama_request.startswith('/') or ollama_request.startswith('http'):
                url = resolve_url(request)
                json = g_client.get_url(url, response_as=str)
                ollama_request = json

            # Send POST request to Ollama API
            ollama_url = urllib.parse.urljoin(config_str('ollama_url'), endpoint)
            _log(f"exec_ollama: POST {ollama_url}:")
            _log(f"{ollama_request[:100]}... ({len(ollama_request)})")
            response = requests.post(ollama_url, data=ollama_request, headers=g_headers_json, timeout=120)
            response.raise_for_status()

            # Parse response
            body = response.json()
            _log(f"exec_ollama response: {body}")

            # Send response to replyTo URL
            g_client.post_url(reply_url, body)
            return

        except requests.exceptions.ConnectionError as e:
            _log("Error: Could not connect to Ollama API. Make sure Ollama is running on localhost:11434")
            error = ResponseStatus(error_code='ConnectionError', message=f"{e or 'Could not connect to Ollama API'}")
        except requests.exceptions.Timeout as e:
            _log("Error: Request timed out. The model might be taking too long to respond.")
            error = ResponseStatus(error_code='Timeout', message=f"{e or 'Request timed out'}")
        except requests.exceptions.RequestException as e:
            error = ResponseStatus(error_code='RequestException', message=f"{e or 'Error making request to Ollama API'}")
        except WebServiceException as e:
            error = e.response_status
        except Exception as e:
            error = ResponseStatus(error_code='Exception', message=f"{e}")

        body = {
            'responseStatus': {
                'errorCode': error.error_code,
                'message': error.message
            }
        }
        _log(f"exec_ollama error: {reply_url} {error.error_code} {error.message}")
        g_client.post_url(reply_url, body)
    except Exception as e:
        _log(f"Error executing Ollama: {e}")
        traceback.print_exc()

def ollama_generate(image_bytes, model, prompt):
    """
    Send an image to Ollama /api/generate API
    """
    # Ollama API endpoint
    url = urllib.parse.urljoin(config_str('ollama_url'), '/api/generate')

    # Encode image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    if not base64_image:
        return None

    # Prepare the request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }

    try:
        # Send POST request to Ollama API
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        # Parse response
        result = response.json()

        if 'response' in result:
            return result['response']
        else:
            _log("Error: No 'response' field in API response")
            _log(f"Full response: {result}")
            return None
    except requests.exceptions.ConnectionError:
        _log("Error: Could not connect to Ollama API. Make sure Ollama is running on localhost:11434")
        return None
    except requests.exceptions.Timeout:
        _log("Error: Request timed out. The model might be taking too long to respond.")
        return None
    except requests.exceptions.RequestException as e:
        _log(f"Error making request: {e}")
        return None
    except json.JSONDecodeError as e:
        _log(f"Error parsing JSON response: {e}")
        return None

def caption_image(artifact_url, model):
    try:
        if g_language_models is None:
            _log(f"caption_image: g_language_models is None {config_str('ollama_url')}")
            return
        if artifact_url is None:
            _log("caption_image: url is None")
            return
        if model is None:
            _log("caption_image: model is None")
            return
        if model not in g_language_models:
            _log(f"caption_image: model {model} is not available")
            return

        url = resolve_url(artifact_url)
        _log(f"caption_image ({model}) GET: {url}")

        image_bytes = url_to_bytes(url)
        if image_bytes is None:
            return

        request = CaptionArtifact(device_id=DEVICE_ID, artifact_url=artifact_url)
        request.caption = ollama_generate(image_bytes, model, "A caption of this image: ")
        request.description = ollama_generate(image_bytes, model, "A detailed description of this image: ")

        _log(f"caption_image caption: {request.caption}\n{request.description}")
        g_client.post(request)

    except Exception as e:
        _log_error("Error captioning image: ", e)
        traceback.print_exc()

def on_prompt_handler(json_data):
    if is_enabled():
        # run send_update once in background thread
        threading.Thread(target=send_update, daemon=True).start()
    return json_data

def try_gpu_infos():
    """
    get info of gpus from $nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader,nounits
    example output: 0, 16303, 13991, 1858
    """
    gpus = []
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'])
    lines = output.decode('utf-8').strip().split('\n')
    for line in lines:
        index, name, total, free, used = line.split(',')
        gpu = GpuInfo(index=int(index),name=name.strip(),total=int(total),free=int(free),used=int(used))
        gpus.append(gpu)
    return gpus

def gpu_infos():
    try:
        return try_gpu_infos()
    except Exception as e:
        _log_error("Error getting GPU info: ", e)
        print(output)
        return []

def gpus_as_jsv():
    gpus = gpu_infos()
    # complex types on the query string need to be sent with JSV format
    ret = ','.join(['{' + f"index:{gpu.index},name:\"{gpu.name}\",total:{gpu.total},free:{gpu.free},used:{gpu.used}" + '}' for gpu in gpus])
    return ret

def register_agent():
    # get workflows from user/default/workflows
    user_dir = get_user_directory()
    workflows_dir = os.path.join(user_dir, "default", "workflows")
    workflows = []
    # exclude .json starting with '.'
    if os.path.exists(workflows_dir):
        workflows = [f for f in os.listdir(workflows_dir) if f.endswith(".json") and not f.startswith(".")]

    object_info_json = get_object_info_json()

    object_info_file = UploadFile(
        field_name="object_info",
        file_name="object_info.json",
        content_type="application/json",
        stream=io.BytesIO(object_info_json.encode('utf-8')))

    global g_language_models
    g_language_models = None
    ollama_base_url = config_str('ollama_url')
    if ollama_base_url:
        try:
            g_language_models = []
            # Check if Ollama is running by hitting the base endpoint with a reasonable timeout
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=10)
            response.raise_for_status()

            # Parse the response
            data = response.json()

            # Extract models from the response
            models = data.get('models', [])

            _log(f"✅ Ollama is running with {len(models)} installed models")
            for i, model in enumerate(models, 1):
                name = model.get('name')
                if name is not None:
                    g_language_models.append(name)
        except requests.exceptions.ConnectionError:
            _log(f"❌ Cannot connect to Ollama at {ollama_base_url}")
            _log("Make sure Ollama is running and accessible")
        except requests.exceptions.Timeout:
            _log(f"❌ Request to {ollama_base_url} timed out")
        except requests.exceptions.RequestException as e:
            _log(f"❌ Error connecting to Ollama: {e}")
        except Exception as e:
            _log(f"❌ Unexpected error: {e}")

    error = None
    try:
        gpus = try_gpu_infos()
    except Exception as e:
        error = e
        gpus = []

    response = g_client.post_file_with_request(
        request=RegisterComfyAgent(
            device_id=DEVICE_ID,
            version=VERSION,
            comfy_version=get_comfyui_version(),
            workflows=workflows,
            gpus=gpus,
            queue_count=get_queue_count(),
            models=get_model_files(),
            language_models=g_language_models,
            installed_pip=g_installed_pip_packages,
            installed_nodes=g_installed_custom_nodes,
            installed_models=g_installed_models,
            config=ComfyAgentConfig(
                install_models=allow_installing_models(),
                install_nodes=allow_installing_nodes(),
                install_packages=allow_installing_packages(),
            )
        ),
        file=object_info_file)

    if error is not None:
        update_status_error(error, "Failed to get GPU info:")

    _log(f"Registered device with {config_str('url')}")
    printdump(response)

    # check if response.categories is an array with items
    if isinstance(response.categories, list):
        global g_categories
        g_categories = response.categories

    if isinstance(response.require_pip, list):
        if len(g_installed_pip_packages) == 0:
            _log(f"Installing required pip packages: {len(response.require_pip)}")
        for package in response.require_pip:
            if package not in g_installed_pip_packages:
                install_pip_package(package)
    if isinstance(response.require_nodes, list):
        if len(g_installed_custom_nodes) == 0:
            _log(f"Installing required custom nodes: {len(response.require_nodes)}")
        for node in response.require_nodes:
            if node not in g_installed_custom_nodes:
                install_custom_node(node)
    if isinstance(response.require_models, list):
        if len(g_installed_models) == 0:
            _log(f"Downloading required models: {len(response.require_models)}")
        for saveto_and_model in response.require_models:
            if saveto_and_model not in g_installed_models:
                install_model(saveto_and_model)
    if isinstance(response.settings, ComfyAgentSettings):
        global g_settings
        g_settings = response.settings

def to_error_status(e: Exception, error_code=None, message=None):
    if e is None:
        return None
    if error_code is None:
        error_code = type(e).__name__ or "Exception"
    if message is None:
        message = f"{e}"
    elif message.endswith(":"):
        message += f" {e}"

    if isinstance(e, subprocess.CalledProcessError):
        return ResponseStatus(
            error_code=error_code,
            message=message,
            stack_trace=e.stderr.decode('utf-8') if e.stderr is not None else traceback.format_exc())
    return ResponseStatus(
        error_code,
        message=message,
        stack_trace=traceback.format_exc())

def update_status_error(e: Exception, msg: str = None):

    error = to_error_status(e, message=msg)

    # if is subprocess.CalledProcessError
    if isinstance(e, subprocess.CalledProcessError):
        _log(msg)
        stdout = e.stdout.decode('utf-8') if e.stdout is not None else ""
        update_status_async(status=msg, logs=f"{stdout}", error=error, wait=0)
        return

    if msg is not None:
        _log(f"{msg}: {e}")
    send_update(status="Error", error=error)

def update_status_async(status: str, logs: str = None, error: ResponseStatus = None, wait=1):
    """
    Update the status, logs or error of the agent asynchronously (without the full update context in send_update).
    """
    _log(f"status: {status}")
    if not is_enabled():
        return
    g_statuses.append(UpdateComfyAgentStatus(
        device_id=DEVICE_ID,
        status=status,
        logs=logs,
        error=error))
    if wait == 0:
        update_status(wait)
    else:
        threading.Thread(target=update_status, args=(wait,), daemon=True).start()

def update_status(wait=1):
    time.sleep(wait)
    if len(g_statuses) > 0:
        last_status = g_statuses.pop()
        g_statuses.clear()
        try:
            g_client.post(last_status)
        except Exception as e:
            _log(f"Error sending update status: {e}")

def send_update_async(status=None, error=None):
    threading.Thread(target=send_update, args=(status, error), daemon=True).start()

def send_update(status=None, error=None):
    try:
        current_queue = PromptServer.instance.prompt_queue.get_current_queue()
        queue_running = current_queue[0]
        queue_pending = current_queue[1]

        request = UpdateComfyAgent(device_id=DEVICE_ID,
            gpus=gpu_infos(),
            models=get_model_files(),
            status=status,
            error=error,
            language_models=g_language_models,
            installed_pip=g_installed_pip_packages,
            installed_nodes=g_installed_custom_nodes,
            installed_models=g_installed_models
        )

        request.queue_count = len(queue_running) + len(queue_pending)

        # get running generation ids (client_id) (max 20)
        request.running_generation_ids = [entry[3]['client_id'] for entry in queue_running
            if len(entry[3]['client_id']) == 32][:20]
        # get queued generation ids (client_id) (max 20)
        request.queued_generation_ids = [entry[3]['client_id'] for entry in queue_pending
            if len(entry[3]['client_id']) == 32][:20]

        _log(f"send_update({status}): queue_count={request.queue_count}, running={request.running_generation_ids}, queued={request.queued_generation_ids}")
        # print(request.installed_nodes)
        g_statuses.clear()
        g_client.post(request)

    except WebServiceException as ex:
        status = ex.response_status
        if status.error_code == "NotFound":
            _log("Device not found, reregistering")
            register_agent()
            return
        else:
            _log(f"Error sending update: {ex.message}\n{printdump(status)}")
    except Exception as e:
        _log_error("Error sending update: ", e)

def save_installed_items(file:str, items:list):
    _log(f"Saving {file}")
    items.sort()
    with open(os.path.join(g_node_dir, file), "w") as f:
        f.write("\n".join(items))

def load_installed_items(file:str):
    install_path = os.path.join(g_node_dir, file)
    if not os.path.exists(install_path):
        return []
    try:
        with open(install_path, "r") as f:
            lines = f.read().splitlines()
            _log(f"Loaded {len(lines)} installed items from {file}")
            printdump(lines)
            return lines
    except Exception as e:
        _log(f"Error loading installed items from {file}: {e}")
        return []

def append_installed_item(file:str, items:list, item:str):
    if item in items:
        return
    items.append(item)
    save_installed_items(file, items)

def assert_can_install(allowed, config_name):
    if not allowed:
        install_type = config_name.replace('_', 'ing ').replace('nodes','custom nodes')
        message = f"{config_name} is disabled. This agent does not allow {install_type}."
        send_update(status=message, error=ResponseStatus(
                error_code="InstallFailed",
                message=message
            ))
        return False
    return True

# pip version_operators = r'(==|>=|<=|>|<|!=|~=|===)'
# pip version_pattern = r'[a-zA-Z0-9._+!-]+'
SHELL_ESCAPE_CHARS = ['\'', '"', '`', '$', '[', ']', ';', '|', '*', '\\', '\t', '\n', '\r']
def assert_no_shell_escape_chars(name):
    if any(c in name for c in SHELL_ESCAPE_CHARS):
        message = "Invalid input contains shell escape characters"
        send_update(status=message, error=ResponseStatus(
                error_code="InstallFailed",
                message=message
            ))
        return False
    return True

def uninstall_pip_package(package_name):
    if not assert_can_install(allow_installing_packages(), "install_packages"):
        return
    if not assert_no_shell_escape_chars(package_name):
        return
    try:
        send_update_async(status=f"Uninstalling {package_name}...")
        o = subprocess.run(['pip', 'uninstall', '-y', package_name], check=True)
        g_installed_pip_packages.remove(package_name)
        save_installed_items("requirements.txt", g_installed_pip_packages)
        send_update(status=f"Uninstalled {package_name}")
        return o
    except Exception as e:
        update_status_error(e, f"Error uninstalling {package_name}")
        return None

def install_pip_package(package_name):
    if not assert_can_install(allow_installing_packages(), "install_packages"):
        return
    if not assert_no_shell_escape_chars(package_name):
        return
    pkg = package_name
    if (package_name.endswith("requirements.txt")):
        # Use directory name as package name
        pkg = os.path.basename(os.path.dirname(package_name)) + "/requirements.txt"

    try:
        send_update_async(status=f"Installing {pkg}...")
        if package_name.endswith("requirements.txt"):
            o = subprocess.run(['pip', 'install', '-r', package_name], check=True)
        else:
            o = subprocess.run(['pip', 'install', package_name], check=True)
            append_installed_item("requirements.txt", g_installed_pip_packages, package_name)
        send_update(status=f"Installed {pkg}. {MSG_RESTART_COMFY}")

        return o
    except Exception as e:
        send_update(error=to_error_status(e, message=f"Error installing {pkg}:"))
        return None

def uninstall_custom_node(url):
    if not assert_can_install(allow_installing_nodes(), "install_nodes"):
        return
    status = None
    error = None
    try:
        custom_nodes_dir = folder_names_and_paths["custom_nodes"][0][0]
        node_file_or_dir = url.rstrip('/').split('/')[-1]
        custom_node_path = os.path.join(custom_nodes_dir, node_file_or_dir)

        if not os.path.exists(custom_node_path):
            status = f"Custom Node '{node_file_or_dir}' does not exist"
            return

        if os.path.isdir(custom_node_path):
            shutil.rmtree(custom_node_path)
        else:
            os.remove(custom_node_path)

        status = f"Deleted custom node '{node_file_or_dir}'. "

        return
    except Exception as e:
        status = None
        error = to_error_status(e, message=f"Error uninstalling custom node '{node_file_or_dir}':")
    finally:
        # remove custom node
        url not in g_installed_custom_nodes or g_installed_custom_nodes.remove(url)
        save_installed_items("require-nodes.txt", g_installed_custom_nodes)
        send_update(status=status, error=error)

def install_custom_node(repo_url):
    if not assert_can_install(allow_installing_nodes(), "install_nodes"):
        return
    try:
        custom_nodes_dir = folder_names_and_paths["custom_nodes"][0][0]

        # URLs ending with '.py' should be copied to custom_nodes directory
        if repo_url.endswith(".py"):
            # download file to custom_nodes directory
            node_filename = repo_url.split('/')[-1]
            custom_node_path = os.path.join(custom_nodes_dir, node_filename)
            if os.path.exists(custom_node_path):
                _log(f"{custom_node_path} already exists")
                append_installed_item("require-nodes.txt", g_installed_custom_nodes, repo_url)
                send_update(status=f"Already downloaded {node_filename}")
                return None
            # use requests to download python file:
            response = requests.get(repo_url)
            with open(custom_node_path, 'wb') as f:
                f.write(response.content)

            append_installed_item("require-nodes.txt", g_installed_custom_nodes, repo_url)
            send_update(status=f"Downloaded {node_filename}. {MSG_RESTART_COMFY}")
            return None

        repo_name = repo_url.split('/')[-1]
        custom_node_path = os.path.join(custom_nodes_dir, repo_name)
        if os.path.exists(custom_node_path):
            _log(f"{custom_node_path} already exists")
            append_installed_item("require-nodes.txt", g_installed_custom_nodes, repo_url)
            send_update(status=f"Already installed {repo_name}")
            return None

        _log("Installing custom node: " + repo_name + " in " + custom_nodes_dir)
        send_update_async(status=f"Installing {repo_name}...")
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]
        # if repo_url does not contain '://' append https://github.com
        if '://' not in repo_url:
            repo_url = urljoin("https://github.com", repo_url)

        o = subprocess.run(['git', 'clone', repo_url, custom_node_path], check=True)

        # if they have a requirements.txt, install it
        if os.path.exists(os.path.join(custom_node_path, "requirements.txt")):
            # logs=f"{o.stdout}"
            send_update_async(status=f"Installing {repo_name} requirements.txt...")
            o = install_pip_package(os.path.join(custom_node_path, "requirements.txt"))

        append_installed_item("require-nodes.txt", g_installed_custom_nodes, repo_url)
        send_update(status=f"Installed {repo_name}. {MSG_RESTART_COMFY}")

        return o
    except Exception as e:
        update_status_error(e, f"Error installing {repo_url}:")
        return None

def delete_model(path):
    if not assert_can_install(allow_installing_models(), "install_models"):
        return
    global g_installed_models
    status = None
    error = None
    try:
        model_path = os.path.join(models_dir, path)
        if not os.path.exists(model_path):
            status = f"Model {path} does not exist"
            return

        os.remove(model_path)
        status = f"Deleted model {path}"

        # delete folder if empty
        if not os.listdir(os.path.dirname(model_path)):
            os.rmdir(os.path.dirname(model_path))
        # also delete parent folder if empty
        if not os.listdir(os.path.dirname(os.path.dirname(model_path))):
            os.rmdir(os.path.dirname(os.path.dirname(model_path)))
    except Exception as e:
        status = None
        error = to_error_status(e, message=f"Error deleting {path}:")
    finally:
        # remove model starting with path
        global g_installed_models
        g_installed_models = [m for m in g_installed_models if not m.startswith(path)]
        save_installed_items("require-models.txt", g_installed_models)
        send_update(status=status, error=error)

def install_model(saveto_and_url:str):
    save_to, url = saveto_and_url.split(' ', 1)
    return download_model(save_to.strip(), url.strip())

def assert_path_within(path, within):
    """
    Check if path is within another path
    """
    if not os.path.commonpath((within, path)) == within:
        send_update(status="Install Failed.", error=ResponseStatus(
            error_code="InvalidPath", message=f"Invalid path, must be within {os.path.basename(within)}"))
        return False
    return True

def download_model(save_to, url, progress_callback=None):
    if not assert_can_install(allow_installing_models(), "install_models"):
        return
    try:
        # _log(f"download_model({save_to}, {url})")
        item = f"{save_to} {url}"

        save_to_path = os.path.join(models_dir, save_to)
        if os.path.exists(save_to_path):
            append_installed_item("require-models.txt", g_installed_models, item)
            send_update(status=f"{save_to_path} already exists")
            return

        # Sanitize save_to to ensure it's within models_dir
        safe_save_to = os.path.normpath(os.path.join(models_dir, save_to))
        if not assert_path_within(safe_save_to, models_dir):
            return

        # create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(save_to_path), exist_ok=True)

        # url format is <env-with-token?>@?<url-to-download> e,g:
        # $GITHUB_TOKEN@https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx
        # $GITHUB_TOKEN@https://raw.githubusercontent.com/notAI-tech/NudeNet/refs/heads/v3/nudenet/320n.onnx
        # $HF_TOKEN@https://huggingface.co/erax-ai/EraX-NSFW-V1.0/resolve/5cb3aace4faa3e42ff6cfeb97fd93c250c65d7fb/erax_nsfw_yolo11m.pt
        # https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/diffusion_models/hidream_i1_fast_fp8.safetensors

        curl_args = ['curl','-L']
        requests_headers = {}
        if '@' in url:
            token, url = url.split('@', 1)
            # if it starts with $, replace with env value
            if token.startswith('$'):
                token_lower = token[1:].lower()
                if token_lower.endswith('token'):
                    if token_lower in g_config:
                        token = g_config[token_lower]
                    else:
                        env_token = os.environ.get(token[1:], '')
                        if not env_token:
                            send_update(status=f"Missing environment variable {token[1:]}",
                                error=ResponseStatus(error_code="Unauthorized", message=f"Token {token} is required"))
                            return None
                        else:
                            token = env_token
                else:
                    _log(f"Warning: {token} does not end with '_TOKEN', ignoring substitution...")

            if 'github.com' in url:
                curl_args += ['-H', f'Authorization: token {token}']
                requests_headers['Authorization'] = f'token {token}'
            else:
                curl_args += ['-H', f'Authorization: Bearer {token}']
                requests_headers['Authorization'] = f'Bearer {token}'

        update_status_async(status=f"Downloading {save_to}...")
        curl_args += ['-o', save_to_path, url]
        # start monitoring download in a background thread
        threading.Thread(target=start_monitoring_download, args=(save_to_path, url, requests_headers, progress_callback), daemon=True).start()
        o = subprocess.run(curl_args, check=True)
        if complete_download(save_to, url):
            return o

    except Exception as e:
        send_update(error=to_error_status(e, message=f"Error downoading {url} to {save_to}:"))
    finally:
        return None

def complete_download(save_to, url):
    global g_downloading_model
    g_downloading_model = None

    item = f"{save_to} {url}"
    save_to_path = os.path.join(models_dir, save_to)

    # check if downloaded file is a JSON error, first if its less than 1kb
    # CivitAI error example:
    # {"error":"Unauthorized","message":"The creator of this asset requires you to be logged in to download it"}
    if os.path.getsize(save_to_path) < MIN_DOWNLOAD_BYTES:
        with open(save_to_path, 'r') as f:
            try:
                # trim
                text = f.read().strip()
                if text.startswith('{'):
                    json_data = json.load(f)
                    if 'message' in json_data:
                        send_update(status=f"Download failed: {json_data['message']}",
                            error=ResponseStatus(
                                error_code=json_data.get('error', 'DownloadFailed'),
                                message=json_data['message']))
                    os.remove(save_to_path)
                    return False

                # if an error, but not a known JSON format, just report the text
                send_update(status=f"Download failed: {text}",
                    error=ResponseStatus(
                        error_code="DownloadFailed",
                        message=text))
            except:
                pass
            finally:
                return False

    filename = os.path.basename(save_to_path)
    send_update(status=f"Downloaded {filename} {format_bytes(os.path.getsize(save_to_path) or 0)}")
    append_installed_item("require-models.txt", g_installed_models, item)
    agent_needs_updating()
    return True

def start_monitoring_download(save_to_path, url, headers, progress_callback):
    global g_downloading_model
    try:
        _log(f"start_monitoring_download({save_to_path}, {url}, {headers})")
        # get content length
        response = requests.head(url, headers=headers, allow_redirects=True)
        _log(f"monitoring_download HEAD {url}, status: {response.status_code}, Content-Length: {response.headers.get('Content-Length')}")
        if response.status_code < 300:
            content_length = int(response.headers.get("Content-Length"))
        else:
            # attempt to get content length from GET
            _log(f"monitoring_download GET {url}")
            response = requests.get(url, headers=headers, stream=True)
            _log(f"monitoring_download GET {url}, status: {response.status_code}, Content-Length: {response.headers.get('Content-Length')}")
            if response.status_code < 300:
                content_length = int(response.headers.get("Content-Length"))
                response.close()
            else:
                content_length = None

        if content_length is not None:
            g_downloading_model = save_to_path
            filename = os.path.basename(save_to_path)
            _log(f"monitoring_download: {filename} ({format_bytes(content_length)})")
            partial_download_exists = os.path.exists(save_to_path)
            if not partial_download_exists:
                retry_times = 5
                while not partial_download_exists and retry_times > 0:
                    retry_times -= 1
                    time.sleep(1)
                    partial_download_exists = os.path.exists(save_to_path)
                if not partial_download_exists:
                    g_downloading_model = None
                    _log(f"{save_to_path} partial download does not exist, exiting...")
                    return

            filename = os.path.basename(save_to_path)
            while g_downloading_model == save_to_path and os.path.exists(save_to_path):
                partial_download_length = os.path.getsize(save_to_path)
                if partial_download_length >= content_length:
                    g_downloading_model = None
                    if partial_download_length > MIN_DOWNLOAD_BYTES:
                        _log(f"Downloaded {filename} ({format_bytes(partial_download_length)})")
                        send_update(status=f"Downloaded {filename} {format_bytes(partial_download_length)}")
                    return

                update_status_async(status=f"Downloading {filename} {format_bytes(partial_download_length)} of {format_bytes(content_length)}...")
                if progress_callback is not None:
                    progress_callback(filename, partial_download_length, content_length)
                time.sleep(2)
        else:
            return
    except Exception as e:
        _log_error(f"Error monitoring download: {url}", e)

def format_bytes(bytes):
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes / 1024:.2f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes / 1024 / 1024:.2f} MB"
    else:
        return f"{bytes / 1024 / 1024 / 1024:.2f} GB"

def reboot():
    reboot_url = f"{get_server_url()}/api/manager/reboot"
    try:
        send_update(status="Rebooting...")

        _log("Rebooting...")
        response = requests.get(reboot_url, timeout=10)
        if response.status_code == 200:
            send_update(status="Rebooted")
        else:
            send_update(status="Reboot failed", error=ResponseStatus(
                error_code="RebootFailed",
                message=f"Reboot failed with status code {response.status_code}"))
    except Exception as e:
        update_status_error(e, f"Error rebooting: {e}")

def filename_list(folder_name):
    try:
        return get_filename_list(folder_name)
    except Exception:
        dir = os.path.join(models_dir, folder_name)
        if (not os.path.exists(dir)):
            return []
        files, folders_all = recursive_search(dir, excluded_dir_names=[".git"])
        return files

def get_model_files():
    models = {}
    for folder in os.listdir(models_dir):
        files = filename_list(folder)
        if len(files) == 0:
            continue
        models[folder] = files
    return models

def get_custom_nodes():
    nodes = []
    return nodes

def get_default_config():
    return {
        "enabled":          DEFAULT_AUTOSTART,
        'install_models':   DEFAULT_INSTALL_MODELS,
        'install_nodes':    DEFAULT_INSTALL_NODES,
        'install_packages': DEFAULT_INSTALL_PACKAGES,
        "apikey": "",
        "url": DEFAULT_ENDPOINT_URL,
        "ollama_url": os.environ.get("OLLAMA_URL") or "",
        "hf_token": os.environ.get("HF_TOKEN") or "",
        "civitai_token": os.environ.get("CIVITAI_TOKEN") or "",
        "github_token": os.environ.get("GITHUB_TOKEN") or ""
    }

def load_config():
    global DEVICE_ID, g_config
    try:
        os.makedirs(g_node_dir, exist_ok=True)
        g_config = get_default_config()

        # Read device ID from users/device-id
        device_id_path = os.path.join(g_node_dir, "device-id")
        # check if file exists
        if os.path.isfile(device_id_path):
            with open(device_id_path) as f:
                DEVICE_ID = f.read().strip()
            _log(f"DEVICE_ID: {DEVICE_ID}")
        else:
            # write device id
            _log(f"Generating Device ID at {device_id_path}")
            DEVICE_ID = uuid.uuid4().hex
            with open(device_id_path, "w") as f:
                f.write(DEVICE_ID)
    except IOError:
        DEVICE_ID = uuid.uuid4().hex
        _log(f"Failed to read device ID from {device_id_path}. Generating a new one: {DEVICE_ID}")

    try:
        _log("Loading config...")
        config_path = os.path.join(g_node_dir, "config.json")
        if not os.path.exists(config_path):
            save_config(g_config)
            return
        with open(config_path, "r") as f:
            g_config = json.load(f)
    except Exception as e:
        _log(f"Error loading config: {e}")

def agent_needs_updating():
    global g_needs_update
    g_needs_update = True

def save_config(config):
    global g_config, g_client
    g_config.update(config)
    g_client = create_client()
    agent_needs_updating()
    os.makedirs(g_node_dir, exist_ok=True)
    _log("Saving config...")
    # _log("Saving config: " + json.dumps(g_config))
    with open(os.path.join(g_node_dir, "config.json"), "w") as f:
        json.dump(g_config, f, indent=4)

def start():
    global g_client, g_running, g_installed_pip_packages, g_installed_custom_nodes, g_installed_models

    if g_running:
        _log("Already running")
        return

    if not config_str("url"):
        _log("No URL configured. Please configure in the ComfyAgentNode.")
        return
    if not config_str("apikey"):
        _log("No API key configured. Please configure in the ComfyAgentNode.")
        return
    if not is_enabled():
        _log("Autostart is disabled. Enable in the ComfyAgentNode.")
        return

    g_running = True
    apikey = config_str("apikey")
    g_client = create_client()

    hidden_token = apikey[:3] + ("*" * 3) + apikey[-2:]
    _log(f"ComfyGateway {hidden_token}@{g_config['url']}")

    # Replace the original method with your interceptor
    PromptServer.send_sync = intercepted_send_sync

    try:
        g_headers["User-Agent"] = g_headers_json["User-Agent"] = f"comfy-agent/{get_comfyui_version()}/{VERSION}/{DEVICE_ID}"

        g_installed_pip_packages = load_installed_items("requirements.txt")
        g_installed_custom_nodes = load_installed_items("require-nodes.txt")
        g_installed_models = load_installed_items("require-models.txt")

        custom_nodes_dir = folder_names_and_paths["custom_nodes"][0][0]
        custom_node_items_lower = [f.lower() for f in os.listdir(custom_nodes_dir)]

        changed = False
        remove_items = []
        for repo_url in g_installed_custom_nodes:
            dir_or_file = repo_url.split("/")[-1]
            if not dir_or_file.lower() in custom_node_items_lower:
                remove_items.append(repo_url)
                _log(f"Removed missing custom node {repo_url}")
                changed = True

        for repo_url in remove_items:
            repo_url not in g_installed_custom_nodes or g_installed_custom_nodes.remove(repo_url)
        # ....
        custom_nodes_lower = list((repo_url.lower() for repo_url in g_installed_custom_nodes))
        for folder in os.listdir(custom_nodes_dir):
            dir_path = os.path.join(custom_nodes_dir, folder)
            if os.path.isdir(dir_path):
                # check if folder has .git folder
                if not os.path.exists(os.path.join(dir_path, '.git')):
                    continue
                try:
                    repo_url = subprocess.check_output(['git', '-C', dir_path, 'config', '--get', 'remote.origin.url']).decode('utf-8').strip()
                    if repo_url.lower() not in custom_nodes_lower:
                        g_installed_custom_nodes.append(repo_url)
                        _log(f"Added custom node {repo_url}")
                        changed = True
                except subprocess.CalledProcessError as e:
                    _log_error(f"Failed to get repo url for {folder}: ", e)
        if changed:
            save_installed_items("require-nodes.txt", g_installed_custom_nodes)

        changed = False
        remove_items = []
        for saveto_and_url in g_installed_models:
            save_to = saveto_and_url.split(" ", 1)[0]
            model_path = os.path.join(models_dir, save_to)
            if not os.path.exists(model_path):
                remove_items.append(saveto_and_url)
                _log(f"Removed missing model {save_to}")
                changed = True

        for saveto_and_url in remove_items:
            saveto_and_url not in g_installed_models or g_installed_models.remove(saveto_and_url)

        if changed:
            save_installed_items("require-models.txt", g_installed_models)

        try:
            _log("Setting up global polling task.")
            # register_agent()
            # listen to messages in a background thread
            t = threading.Thread(target=listen_to_messages_poll, daemon=True)
            t.start()

        except Exception:
            logging.error("[ERROR] Could not connect to ComfyGateway.")
            logging.error(traceback.format_exc())
    except Exception:
        logging.error("[ERROR] Could not load models.")
        logging.error(traceback.format_exc())

def update_agent(config):
    global g_client, g_config, g_needs_update
    save_config(config)
    if is_enabled() and not g_running:
        start()

# --- ComfyUI Node Definition ---
class ComfyAgentNode:
    NODE_NAME = "ComfyAgentNode"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "updated"
    OUTPUT_NODE = True
    CATEGORY = "comfy_agent"

    @classmethod
    def INPUT_TYPES(cls):
        # Use global defaults for the node's default inputs
        # load_config()
        return {
            "required": {
                "enabled":          ("BOOLEAN", {"default": is_enabled(),                "label": "Enabled", "label_on": "YES",   "label_off": "NO"}),
                "apikey":           ("STRING",  {"default": config_str("apikey")}),
                "install_models":   ("BOOLEAN", {"default": allow_installing_models(),   "label": "Enabled", "label_on": "ALLOW", "label_off": "DENY"}),
                "install_nodes":    ("BOOLEAN", {"default": allow_installing_nodes(),    "label": "Enabled", "label_on": "ALLOW", "label_off": "DENY"}),
                "install_packages": ("BOOLEAN", {"default": allow_installing_packages(), "label": "Enabled", "label_on": "ALLOW", "label_off": "DENY"}),
                "url":              ("STRING",  {"default": config_str("url")}),
            },
            "optional": {
                "ollama_url":       ("STRING",  {"default": config_str("ollama_url")}),
                "hf_token":         ("STRING",  {"default": config_str("hf_token")}),
                "civitai_token":    ("STRING",  {"default": config_str("civitai_token")}),
                "github_token":     ("STRING",  {"default": config_str("github_token")}),
            }
        }
    #"trigger_restart": ("*",),

    def __init__(self):
        self._node_log_prefix_str = f"[{self.NODE_NAME} id:{hex(id(self))[-4:]}]"
        _log("Node instance initialized. This node controls the global polling task.")

    def updated(self, enabled, apikey, install_models, install_nodes, install_packages,
                url, ollama_url, hf_token, civitai_token, github_token):
        update_agent({
            "enabled": enabled,
            "apikey": apikey,
            "install_models": install_models,
            "install_nodes": install_nodes,
            "install_packages": install_packages,
            "url": url,
            "ollama_url": ollama_url,
            "hf_token": hf_token,
            "civitai_token": civitai_token,
            "github_token": github_token,
        })
        _log(f"Node updated. Enabled: {enabled}, URL: {url}")

        return ()

class RegisterComfyAgentNode:
    # --- ComfyUI Registration ---
    NODE_CLASS_MAPPINGS = {
        ComfyAgentNode.NODE_NAME: ComfyAgentNode
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        ComfyAgentNode.NODE_NAME: "Comfy Agent (Global)"
    }

# --- Autostart Logic ---

load_config()
PromptServer.instance.add_on_prompt_handler(on_prompt_handler)

# Check COMFY_GATEWAY environment variable for BASE_URL and BEARER_TOKEN configuration:
# BEARER_TOKEN@BASE_URL
COMFY_GATEWAY = os.environ.get('COMFY_GATEWAY')
if COMFY_GATEWAY:
    if "@" in COMFY_GATEWAY:
        bearer_token, base_url = COMFY_GATEWAY.split("@")
        save_config({
            "url": base_url,
            "apikey": bearer_token,
        })
    else:
        _log(f"Warning: COMFY_GATEWAY environment variable is not in the correct format. Expected 'BEARER_TOKEN@BASE_URL', got '{COMFY_GATEWAY}'.")

start()
