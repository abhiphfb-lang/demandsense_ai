"""
Entry point: launches the Streamlit app (streamlit_app.py) in this folder.

Usage:
    python main.py
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(BASE_DIR, "streamlit_app.py")


def run():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", APP_PATH],
        cwd=BASE_DIR,
    )


if __name__ == "__main__":
    run()
