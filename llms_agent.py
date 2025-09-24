#!/usr/bin/env python

import os
import threading
import time
import json
import uuid
import logging
import traceback
import asyncio
import aiohttp

# from servicestack import JsonServiceClient, ResponseStatus
from .utils import Paths
from .llms import chat_completion, chat_summary, init_llms, load_llms

from folder_paths import (
    base_path, get_user_directory, models_dir
)

g_running = False
g_logger_prefix = ""
g_config = {'enabled': False}
g_llms_config = {'enabled': False}
g_device=uuid.uuid4().hex
g_agent = None
g_paths = Paths(base = base_path,
    models = models_dir,
    user = get_user_directory())

def _log(message):
    """Helper method for logging from the global polling task."""
    print(f"{g_logger_prefix} {message}", flush=True)

def is_enabled():
    global g_config
    return 'enabled' in g_config and g_config['enabled'] or False

def load_device():
    global g_device
    os.makedirs(g_paths.agent, exist_ok=True)
    device_id_path = os.path.join(g_paths.agent, "device-id")
    # check if file exists
    if os.path.isfile(device_id_path):
        with open(device_id_path) as f:
            g_device = f.read().strip()
        _log(f"DEVICE_ID: {g_device}")
    else:
        # write device id
        _log(f"Generating Device ID at {device_id_path}")
        g_device = uuid.uuid4().hex
        with open(device_id_path, "w") as f:
            f.write(g_device)

def load_config(agent):
    global g_config, g_logger_prefix, g_device
    try:
        if agent is not None:
            g_logger_prefix = f"[{agent}]"

        config_path = os.path.join(g_paths.agent, "llms_agent.json")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                _log(f"Creating default llms_agent.json at {config_path}")
                copy_default_path = os.path.join(os.path.dirname(__file__), "defaults", "llms_agent.json")
                with open(copy_default_path, "r") as f2:
                    g_config = json.load(f2)
                f.write(json.dumps(g_config, indent=4))
        else:
            with open(config_path, "r") as f:
                g_config = json.load(f)
    except Exception as e:
        _log(f"Error loading config: {e}")

def load_llms_config():
    global g_llms_config
    try:
        config_path = os.path.join(g_paths.agent, "llms.json")

        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                _log(f"Creating default llms.json at {config_path}")
                copy_default_path = os.path.join(os.path.dirname(__file__), "defaults", "llms.json")
                with open(copy_default_path, "r") as f2:
                    g_llms_config = json.load(f2)
                f.write(json.dumps(g_llms_config, indent=4))
        else:
            with open(config_path, "r") as f:
                g_llms_config = json.load(f)
        return g_llms_config
    except Exception as e:
        _log(f"Error loading gateway config: {e}")

def save_config(config):
    global g_config
    g_config.update(config)
    os.makedirs(g_paths.agent, exist_ok=True)
    _log("Saving config...")
    # _log("Saving config: " + json.dumps(g_config))
    with open(os.path.join(g_paths.agent, "llms_agent.json"), "w") as f:
        json.dump(g_config, f, indent=4)

def config_str(name:str):
    return name in g_config and g_config[name] or ""

def update_agent(config):
    global g_client
    update_config(config)
    if is_enabled() and not g_running:
        start()

# --- ComfyUI Node Definition ---
class LLMsAgentNode:
    NODE_NAME = "LLMsAgentNode"
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
                "enabled": ("BOOLEAN", {"default": is_enabled(), "label": "Enabled", "label_on": "YES", "label_off": "NO"}),
                "apikey":  ("STRING",  {"default": config_str("apikey")}),
                "url":     ("STRING",  {"default": config_str("url")}),
            }
        }
    #"trigger_restart": ("*",),

    def __init__(self):
        self._node_log_prefix_str = f"[{self.NODE_NAME} id:{hex(id(self))[-4:]}]"
        # _log("Node instance initialized. This node controls the global polling task.")

    def updated(self, enabled, apikey, url):
        update_agent({
            "enabled": enabled,
            "apikey": apikey,
            "url": url,
        })
        _log(f"Node updated. Enabled: {enabled}, URL: {url}")

        return ()

class RegisterLLMsAgentNode:
    # --- ComfyUI Registration ---
    NODE_CLASS_MAPPINGS = {
        LLMsAgentNode.NODE_NAME: LLMsAgentNode
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        LLMsAgentNode.NODE_NAME: "LLMs Agent"
    }


def update_config(config):
    global g_client
    save_config(config)

def start_polling():
    asyncio.run(listen_to_messages_poll())

async def listen_to_messages_poll(sleep=3):
    global g_client, g_running, g_needs_update
    g_running = True
    # g_client = create_client()

    base_url = config_str('url')
    models = g_config['models']
    api_key = config_str('apikey')
    headers = g_config['headers'].copy()
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    await load_llms()

    retry_secs = 5
    time.sleep(sleep)

    models_str = ','.join(models)
    url = f"{base_url}/api/GetChatCompletion?device={g_device}&models={models_str}"

    async with aiohttp.ClientSession() as session:
        while is_enabled():
            try:
                _log(f"Polling for chat requests: GET {url}")
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as task_res:
                    task_res.raise_for_status()
                    chat_request = await task_res.json() if task_res.status == 200 else None # Ignore Empty 204 Responses
                    if chat_request:
                        _log(f"Chat request: {chat_summary(chat_request)}")
                        metadata = chat_request['metadata'] or {}
                        reply_to = metadata['replyTo']
                        del metadata['replyTo']
                        if len(metadata) == 0:
                            del chat_request['metadata']
                        response = await chat_completion(chat_request)
                        _log(f"Complete chat request: POST {reply_to}")
                        _log(json.dumps(response, indent=2))
                        async with session.post(reply_to, headers=headers, json=response, timeout=aiohttp.ClientTimeout(total=60)) as response_res:
                            response_res.raise_for_status()
                    else:
                        _log("No pending chat requests, waiting 5s...")
                        await asyncio.sleep(5)
                    retry_secs = 5
            except Exception as e:
                _log(f"Error connecting to {base_url}: {e}, retrying in {retry_secs}s")
                traceback.print_exc()
                _log(f"Waiting {retry_secs}s before retrying...")
                await asyncio.sleep(retry_secs)
                if retry_secs < 5 * 60:
                    retry_secs += 5

    _log(f"Disconnected from {base_url}")
    g_running = False

def start():
    global g_client, g_running

    _log("Loading config...")
    load_device()
    load_config(agent="llms-agent")
    init_llms(load_llms_config())

    if not is_enabled():
        _log("LLMs Agent is disabled. Enable in LLMsAgentNode.")
        return
    if g_running:
        _log("Already running")
        return
    try:
        _log("Setting up global polling task.")
        # register_agent()
        # listen to messages in a background thread, wait for 2 seconds to give custom nodes time to load
        t = threading.Thread(target=start_polling, daemon=True)
        t.start()

    except Exception:
        logging.error("[ERROR] Could not connect to ComfyGateway.")
        logging.error(traceback.format_exc())

start()
