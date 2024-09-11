"""This script auto-generates markdown files from Python docstrings using lazydocs
and tweaks the output for
- prettier badges linking to source code on GitHub
- remove bold tags since they break inline code.
"""

from __future__ import annotations

import json
import os
import subprocess
from glob import glob

ROOT = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT)
with open("site/package.json") as file:
    pkg = json.load(file)
route = "site/src/routes/api"

for path in glob(f"{route}/*.md"):
    os.remove(path)

subprocess.run(
    f"lazydocs {pkg['name']} --output-path {route} "
    f"--no-watermark --src-base-url {pkg['repository']}/blob/main",
    shell=True,
    check=True,
)

for path in glob(f"{route}/*.md"):
    with open(path) as file:
        markdown = file.read()

    # remove all files with less than 20 lines
    # these correspond to mostly empty __init__.py files
    min_line_cnt = 20
    if markdown.count("\n") < min_line_cnt:
        os.remove(path)
        continue

    # remove <b> tags from generated markdown as they break inline code
    markdown = markdown.replace("<b>", "").replace("</b>", "")
    # improve style of badges linking to source code on GitHub
    markdown = markdown.replace(
        'src="https://img.shields.io/badge/-source-cccccc?style=flat-square"',
        'src="https://img.shields.io/badge/source-blue?style=flat" alt="source link"',
    )
    # remove "Global Variables" section if only contains TYPE_CHECKING
    markdown = markdown.replace(
        "\n**Global Variables**\n---------------\n- **TYPE_CHECKING**\n\n", ""
    )
    with open(path, mode="w") as file:
        file.write(markdown)
