"""HekmatAi: agent (hekmatai.agent), own benchmark (hekmatai.own_benchmark),
external reference (hekmatai.arc_reference).

Dependency rule (checked by tests/test_separation.py):
  own_benchmark -> agent, arc_reference -> agent, never the other way and
  never own_benchmark <-> arc_reference.
"""

__version__ = "0.0.1"
