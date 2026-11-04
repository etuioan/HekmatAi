"""The HekmatAi agent. Knows no benchmark; see interface.py."""

from .config import AgentConfig
from .variants import VARIANTS
from .version import agent_version

__all__ = ["AgentConfig", "VARIANTS", "agent_version"]
