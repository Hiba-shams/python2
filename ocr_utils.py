import easyocr
import re
from difflib import get_close_matches
import string

reader = easyocr.Reader(['en'])

def normalize_text(text):
    """Normalize text by stripping, lowercasing, and removing punctuation."""
    text = text.lower().strip()
    return text.translate(str.maketrans('', '', string.punctuation))

def extract_text(image_path):
    """Extract text from the image using EasyOCR."""
    return reader.readtext(image_path, detail=0)

def fuzzy_find_label(labels, line, cutoff=0.7):
    """Find the best matching label using fuzzy matching."""
    line = normalize_text(line)
    norm_labels = [normalize_text(l) for l in labels]
    matches = get_close_matches(line, norm_labels, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def validate_license_fields(text_lines):
    """Validate and extract relevant license fields from OCR text."""
    joined_text = ' '.join(text_lines)

    patterns = {
        "license_number": r"[A-Z]\d{3}-\d{4}-\d{4}",
        "dob": r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b",
        "date": r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b"
    }

    possible_labels = {
        "license_number": ["license no", "lic no", "license number"],
        "name": ["name", "full name"],
        "dob": ["date of birth", "dob", "birth date"],
        "expiry_date": ["expiry date", "exp", "expires"],
        "issue_date": ["issue date", "issued"]
    }

    results = {
        "license_number": None,
        "dob": None,
        "expiry_date": None,
        "issue_date": None,
        "name": None
    }

    for idx, line in enumerate(text_lines):
        norm_line = normalize_text(line)

        # License Number
        if fuzzy_find_label(possible_labels["license_number"], line):
            match = re.search(patterns["license_number"], line)
            if match:
                results["license_number"] = match.group(0)

        # DOB
        if fuzzy_find_label(possible_labels["dob"], line):
            if idx + 1 < len(text_lines):
                next_line = text_lines[idx + 1]
                match = re.search(patterns["date"], next_line)
                if match:
                    results["dob"] = match.group(0)

        # Expiry Date (must be in the next line)
        if fuzzy_find_label(possible_labels["expiry_date"], line):
            if idx + 1 < len(text_lines):
                next_line = text_lines[idx + 1]
                match = re.search(patterns["date"], next_line)
                if match:
                    results["expiry_date"] = match.group(0)

        # Issue Date (must be in the next line)
        if fuzzy_find_label(possible_labels["issue_date"], line):
            if idx + 1 < len(text_lines):
                next_line = text_lines[idx + 1]
                match = re.search(patterns["date"], next_line)
                if match:
                    results["issue_date"] = match.group(0)

        # Name Extraction - from next line after 'name' label
        if fuzzy_find_label(possible_labels["name"], line):
            if idx + 1 < len(text_lines):
                next_line = text_lines[idx + 1].strip()
                if re.match(r'^[A-Za-z\s\.,\-;]+$', next_line):
                    cleaned = next_line.title().strip(" ;,")
                    if "Driver" not in cleaned and "USA" not in cleaned:
                        results["name"] = cleaned
                        continue

    # Fallbacks
    if results["license_number"] is None:
        match = re.search(patterns["license_number"], joined_text)
        if match:
            results["license_number"] = match.group(0)

    if results["dob"] is None:
        match = re.search(patterns["date"], joined_text)
        if match:
            results["dob"] = match.group(0)

    if results["expiry_date"] is None:
        all_dates = re.findall(patterns["date"], joined_text)
        if len(all_dates) >= 2:
            results["expiry_date"] = all_dates[-1]

    if results["issue_date"] is None:
        all_dates = re.findall(patterns["date"], joined_text)
        if len(all_dates) >= 2:
            results["issue_date"] = all_dates[-2]

    # Optional name fallback
    if results["name"] is None:
        for i in range(len(text_lines) - 1):
            candidate = f"{text_lines[i].strip()} {text_lines[i+1].strip()}"
            candidate_clean = re.sub(r'[^A-Za-z\s\-]', '', candidate)
            words = candidate_clean.strip().split()
            if 1 < len(words) <= 3 and all(w.istitle() or w.isupper() for w in words):
                if "DRIVERLICENSE" not in candidate.upper() and "USA" not in candidate.upper():
                    results["name"] = ' '.join(words)
                    break

    is_valid = sum(results[k] is not None for k in ["license_number", "dob", "expiry_date", "name"]) >= 3

    return is_valid, results
