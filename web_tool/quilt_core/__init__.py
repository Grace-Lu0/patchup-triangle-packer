"""Core library for the Ralli quilt web tool.

Data flows one way:
    inventory (harvested triangles)
      -> QuiltPlan (which colour category goes in each slot of each block)
      -> Assignment (which real triangle fills each slot)
      -> rendered image (flat colour or real fabric texture)

Nothing in this package writes files unless a function explicitly says so.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_TOOL_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
