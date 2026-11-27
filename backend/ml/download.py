"""Download NHANES 2007-2018 and the 2019 public-use linked mortality files."""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

CYCLES = {"E": 2007, "F": 2009, "G": 2011, "H": 2013, "I": 2015, "J": 2017}
FILES = ("DEMO", "BMX", "BPX", "TCHOL", "SLQ", "SMQ", "ALQ", "PAQ", "DBQ", "DIQ", "BPQ", "MCQ")
SURVEY = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}"
MORTALITY = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/{name}"
CHECKSUMS = Path(__file__).with_name("checksums.json")


def sources() -> dict[str, str]:
    urls: dict[str, str] = {}
    for suffix, year in CYCLES.items():
        for stem in FILES:
            name = f"{stem}_{suffix}.xpt"
            urls[name] = SURVEY.format(year=year, name=name)
        name = f"NHANES_{year}_{year + 1}_MORT_2019_PUBLIC.dat"
        urls[name] = MORTALITY.format(name=name)
    return urls


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    expected: dict[str, str] = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    for name, url in sources().items():
        target = directory / name
        if not target.exists():
            print(f"downloading {name}")
            partial = target.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=300) as response:
                partial.write_bytes(response.read())
            partial.rename(target)
        if name in expected and sha256(target) != expected[name]:
            sys.exit(f"checksum mismatch for {name}: CDC may have revised the file")
    if not expected:
        CHECKSUMS.write_text(json.dumps({n: sha256(directory / n) for n in sources()}, indent=2) + "\n")


if __name__ == "__main__":
    download(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
