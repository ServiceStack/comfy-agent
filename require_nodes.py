import os
import requests
import shutil
from tqdm import tqdm
from server import PromptServer
from folder_paths import models_dir
from .comfy_agent_node import download_model, install_custom_node, install_pip_package

g_logger_prefix = "[comfy-agent/requires]"
def _log(message):
    print(f"{g_logger_prefix} {message}")

def model_folders():
    return immediate_folders(models_dir)

def immediate_folders(directory_path):
    """
    Returns a list of folders up to 2 levels deep in the specified directory.
    """
    folders = []
    for d in os.listdir(directory_path):
        if os.path.isdir(os.path.join(directory_path, d)):
            name = os.path.basename(d)
            if name.startswith('.'):
                continue
            folders.append(d)
            for sd in os.listdir(os.path.join(directory_path, d)):
                if os.path.isdir(os.path.join(directory_path, d, sd)):
                    name = os.path.basename(sd)
                    if name.startswith('.'):
                        continue
                    folders.append(os.path.join(d, sd))
    return folders

class RequiresAssetNode:
    NODE_NAME = "RequiresAsset"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "download"
    CATEGORY = "comfy_agent"

    def __init__(self):
        self.status = "Idle"
        self.progress = 0.0
        self.node_id = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False, "default": ""}),
                "save_to": (model_folders(), { "default": "checkpoints" }),
                "filename": ("STRING", {"multiline": False, "default": "sdxl_lightning_4step.safetensors"}),
            },
            "optional": {
                "token": ("STRING", { "default": "", "multiline": False, "password": True }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }
    def download(self, url, save_to, filename, node_id, token=""):
        if token:
            url = token + '@' + url
        with tqdm(total=1, unit='iB', unit_scale=True, desc=filename) as pbar:
            download_model(os.path.join(save_to, filename), url,
                progress_callback=lambda file_name, partial_size, total_size:
                    self.update_progress(node_id, pbar, partial_size, total_size))
        return ()

    def update_progress(self, node_id, pbar, partial_size, total_size):
        progress = partial_size/total_size
        pbar.update(progress)
        PromptServer.instance.send_sync("progress", {
            "node": node_id,
            "value": progress,
            "max": 1
        })

class RequiresCustomNode:
    NODE_NAME = "RequiresCustomNode"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "download"
    CATEGORY = "comfy_agent"

    def __init__(self):
        self.status = "Idle"
        self.progress = 0.0
        self.node_id = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    def download(self, repo, node_id):
        # prepend https://github.com/ if not already present
        if not repo.startswith("http"):
            repo = f"https://github.com/{repo}"
        install_custom_node(repo)
        return ()

class RequiresPipPackage:
    NODE_NAME = "RequiresPipPackage"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "download"
    CATEGORY = "comfy_agent"

    def __init__(self):
        self.status = "Idle"
        self.progress = 0.0
        self.node_id = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "package": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    def download(self, package, node_id, version=""):
        install_pip_package(package)
        return ()

# --- ComfyUI Registration ---
class RegisterRequireNodes:
    NODE_CLASS_MAPPINGS = {
        RequiresAssetNode.NODE_NAME: RequiresAssetNode,
        RequiresCustomNode.NODE_NAME: RequiresCustomNode,
        RequiresPipPackage.NODE_NAME: RequiresPipPackage,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        RequiresAssetNode.NODE_NAME: "Requires Asset",
        RequiresCustomNode.NODE_NAME: "Requires Custom Node",
        RequiresPipPackage.NODE_NAME: "Requires PIP Package",
    }
