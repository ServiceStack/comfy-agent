from .comfy_agent_node import RegisterComfyAgentNode
from .require_nodes import RegisterRequireNodes
from .llms_agent import RegisterLLMsAgentNode

NODE_CLASS_MAPPINGS = {
    **RegisterComfyAgentNode.NODE_CLASS_MAPPINGS,
    **RegisterRequireNodes.NODE_CLASS_MAPPINGS,
    **RegisterLLMsAgentNode.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **RegisterComfyAgentNode.NODE_DISPLAY_NAME_MAPPINGS,
    **RegisterRequireNodes.NODE_DISPLAY_NAME_MAPPINGS,
    **RegisterLLMsAgentNode.NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
