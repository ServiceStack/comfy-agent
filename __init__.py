import os
from .comfy_agent_node import RegisterComfyAgentNode
from .require_nodes import RegisterRequireNodes

NODE_CLASS_MAPPINGS = {
    **RegisterComfyAgentNode.NODE_CLASS_MAPPINGS,
    **RegisterRequireNodes.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **RegisterComfyAgentNode.NODE_DISPLAY_NAME_MAPPINGS,
    **RegisterRequireNodes.NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY"
]
