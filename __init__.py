import os
from .comfy_agent_node import RegisterComfyAgentNode
from .asset_downloader import RegisterAssetDownloader

NODE_CLASS_MAPPINGS = {
    **RegisterComfyAgentNode.NODE_CLASS_MAPPINGS,
    **RegisterAssetDownloader.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **RegisterComfyAgentNode.NODE_DISPLAY_NAME_MAPPINGS,
    **RegisterAssetDownloader.NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY"
]
