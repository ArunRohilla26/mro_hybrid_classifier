import re
from typing import List

# Common abbreviation expansions for normalization
ABBREV_MAP = {
    "brg": "bearing",
    "brng": "bearing",
    "vlv": "valve",
    "valv": "valve",
    "pmp": "pump",
    "motar": "motor",
    "motr": "motor",
    "mtr": "motor",
    "assy": "assembly",
    "qty": "quantity",
    "elec": "electrical",
    "mech": "mechanical",
    "wldr": "welder",
    "wldg": "welding",
    "resp": "respirator",
    "extng": "extinguisher",
    "trlly": "trolley",
    "cabnit": "cabinet",
    "noetbook": "notebook",
    "routwr": "router",
    "switc": "switch",
}

def normalize_text(text: str) -> str:
    """Clean and standardize item descriptions for model embedding."""
    if not isinstance(text, str):
        return ""

    s = text.lower()

    # Replace common abbreviations
    for abbr, full in ABBREV_MAP.items():
        s = re.sub(rf"\b{re.escape(abbr)}\b", full, s)

    # Normalize quotes and units
    # Replace "inch" or '"' with ' in'
    s = re.sub(r'["“”]', ' in', s)
    s = s.replace(" inch", " in")

    # Fix "mm" spacing inconsistencies
    s = re.sub(r'\s*mm\b', 'mm', s)

    # Remove unwanted punctuation except for hyphens and slashes
    s = re.sub(r"[^a-z0-9\-/.\s]", " ", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


# Example test (safe to remove in production)
if __name__ == "__main__":
    examples = [
        'SS 10 inch valve',
        'Brg 25mm motr',
        'LED tube light 4 “',
        'Extng cabinet',
        'Resp assy 5 inch'
    ]
    for ex in examples:
        print(f"{ex} -> {normalize_text(ex)}")
