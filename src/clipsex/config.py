from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    data_dir: Path
    index_dir: Path
