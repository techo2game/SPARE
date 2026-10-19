"""Make the PARENet submodule importable.

`pareconv` comes from the PARENet package root, and the unmodified upstream
`backbone`, `model` and `loss` modules come from its 3DMatch experiment. This
directory keeps priority so its `config` and `dataset` are used instead of the
3DMatch ones.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENET_PATH = os.environ.get(
    'SPARE_PARENET_PATH',
    os.path.join(os.path.dirname(HERE), 'PARENet'),
)

for path in (HERE, os.path.join(PARENET_PATH, 'experiments', '3DMatch'), PARENET_PATH):
    if path in sys.path:
        sys.path.remove(path)
sys.path[0:0] = [HERE, os.path.join(PARENET_PATH, 'experiments', '3DMatch'), PARENET_PATH]
