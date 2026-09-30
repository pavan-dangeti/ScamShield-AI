import os
import re
import json
from typing import Dict, List
# Unicode ranges for major Indian scripts
UNICODE_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Punjabi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
}
# Mapping script name to native language guess
SCRIPT_TO_LANG = {
    "Devanagari": "Hindi",
    "Bengali": "Bengali",
    "Tamil": "Tamil",
    "Telugu": "Telugu",
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
    "Gujarati": "Gujarati",
    "Punjabi": "Punjabi",
    "Odia": "Odia",
}
class ScriptLanguageDetector:
    def __init__(self, lang_packs_dir: str = "data/language_packs"):
        self.lang_packs_dir = lang_packs_dir
        self.romanized_dicts: Dict[str, List[str]] = {}
        self.load_language_packs()
    def load_language_packs(self):
        """Loads romanized word lists from language packs."""
        if not os.path.exists(self.lang_packs_dir):
            return
        for file in os.listdir(self.lang_packs_dir):
            if file.endswith(".json") and not file.startswith("explanations"):
                path = os.path.join(self.lang_packs_dir, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        lang_name = data.get("language_name")
                        rom_words = [w.lower() for w in data.get("romanized_words", [])]
                        if lang_name and rom_words:
                            self.romanized_dicts[lang_name] = rom_words
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    def detect(self, text: str) -> Dict[str, str]:
        """
        Detects script type (native/romanized/latin), language guess, and detected script.
        """
        if not text or not isinstance(text, str):
            return {"script_type": "latin", "language_guess": "English", "detected_script": "Latin"}
        
        # Count Unicode blocks
        script_counts = {script: 0 for script in UNICODE_RANGES}
        latin_count = 0
        total_chars = 0
        for char in text:
            code = ord(char)
            matched = False
            for script, (start, end) in UNICODE_RANGES.items():
                if start <= code <= end:
                    script_counts[script] += 1
                    matched = True
                    break
            if not matched:
                if (65 <= code <= 90) or (97 <= code <= 122):  # A-Z, a-z
                    latin_count += 1
            if not char.isspace():
                total_chars += 1
        # Determine if native script predominates
        max_script = None
        max_count = 0
        for script, count in script_counts.items():
            if count > max_count:
                max_count = count
                max_script = script
        # If we have a significant native script character count (at least 3 characters or 5% of text)
        if max_script is not None and max_count > 0 and (max_count >= 3 or (max_count / max(1, total_chars)) > 0.05):
            return {
                "script_type": "native",
                "language_guess": SCRIPT_TO_LANG.get(max_script, "Other"),
                "detected_script": max_script
            }
        # Otherwise, check for romanized code-mixed text
        # Clean text for token matching (keeping words, numbers, and basic punctuation like UPI)
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        lang_matches = {lang: 0 for lang in self.romanized_dicts}
        for token in tokens:
            for lang, words in self.romanized_dicts.items():
                if token in words:
                    lang_matches[lang] += 1
        best_lang = "English"
        best_count = 0
        for lang, count in lang_matches.items():
            if count > best_count:
                best_count = count
                best_lang = lang
        # If we matched romanized words, flag it as romanized
        if best_count > 0:
            return {
                "script_type": "romanized",
                "language_guess": best_lang,
                "detected_script": "Latin"
            }
        else:
            return {
                "script_type": "latin",
                "language_guess": "English",
                "detected_script": "Latin"
            }
if __name__ == "__main__":
    # Test script execution
    det = ScriptLanguageDetector()
    print("Devanagari:", det.detect("कृपया तुरंत भुगतान करें।"))
    print("Tanglish:", det.detect("Unga account block aayidum, pay pannunga link click panni."))
    print("English:", det.detect("Your bank account statement is ready. Download from our secure link."))
