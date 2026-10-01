import csv
import json
import os
import random
import uuid
from typing import Any, Dict, List

# Placeholders for dynamic content generation
BANKS = ["SBI", "HDFC", "ICICI", "Axis Bank", "PNB", "Paytm Bank"]
AMOUNTS = ["₹5,000", "₹10,000", "₹25,000", "₹1,999", "₹15,499", "₹9,999", "₹4,999"]
PHONES = ["+91 9876543210", "8827710293", "9090808070", "7788990011", "9123456789"]
LINKS = [
    "http://bit.ly/pay-upi-ref", 
    "https://t.ly/rbi-kyc-verify", 
    "http://paytm-refund-support.in/claim",
    "https://tinyurl.com/sbi-rewards-points",
    "http://electricity-bill-pay.com",
    "https://kyc-update-verification.com/login"
]
# Language-specific placeholders for time urgency to make text sound natural
LANG_URGENCY_TIMES = {
    "English": ["10 minutes", "30 mins", "24 hours", "tonight by 9:30 PM", "today"],
    "Hindi": ["10 minutes", "30 mins", "24 ghante", "aaj raat", "turant"],
    "Tamil": ["10 minutes", "30 mins", "24 maninerathil", "aaj", "iniku", "udanae"],
    "Telugu": ["10 minutes", "30 mins", "24 gantala lo", "eeroje", "ventane"],
    "Bengali": ["10 minutes", "30 mins", "24 ghontar moddhe", "aajke", "shigghri"]
}
def get_placeholders(lang: str) -> Dict[str, str]:
    times = LANG_URGENCY_TIMES.get(lang, LANG_URGENCY_TIMES["English"])
    return {
        "BANK": random.choice(BANKS),
        "AMOUNT": random.choice(AMOUNTS),
        "PHONE": random.choice(PHONES),
        "LINK": random.choice(LINKS),
        "TIME": random.choice(times),
        "OTP": str(random.randint(1000, 999999))
    }
def fill_template(template: str, placeholders: Dict[str, str]) -> str:
    text = template
    for k, v in placeholders.items():
        text = text.replace(f"{{{k}}}", v)
    return text
class DatasetGenerator:
    def __init__(self, lang_packs_dir: str = "data/language_packs", output_path: str = "data/scam_dataset.csv"):
        self.lang_packs_dir = lang_packs_dir
        self.output_path = output_path
        self.dataset: List[Dict[str, Any]] = []
    def load_language_packs(self) -> List[Dict[str, Any]]:
        packs = []
        if not os.path.exists(self.lang_packs_dir):
            raise FileNotFoundError(f"Language packs directory not found at {self.lang_packs_dir}")
        
        for file in os.listdir(self.lang_packs_dir):
            if file.endswith(".json") and not file.startswith("explanations"):
                path = os.path.join(self.lang_packs_dir, file)
                with open(path, "r", encoding="utf-8") as f:
                    packs.append(json.load(f))
        return packs
    def generate(self):
        packs = self.load_language_packs()
        print(f"Loaded {len(packs)} language packs. Starting generation...")
        for pack in packs:
            lang = pack["language_name"]
            tactic_templates = pack["tactic_templates"]
            legit_templates = pack["legit_templates"]
            # Scripts to generate
            # English is only latin, others have native and romanized
            scripts = ["native"]
            if pack.get("romanized_words") or lang != "English":
                scripts.append("romanized")
            for script in scripts:
                # Get templates that exist for this script
                # (Some language packs might not have romanized if they are English)
                active_tactic_templates = {}
                for tactic, script_dict in tactic_templates.items():
                    if script in script_dict and script_dict[script]:
                        active_tactic_templates[tactic] = script_dict[script]
                active_legit_templates = legit_templates.get(script, [])
                if not active_legit_templates and script == "romanized" and "native" in legit_templates:
                    # Fallback to native legit templates if romanized are missing (e.g. for English)
                    active_legit_templates = legit_templates["native"]
                # Skip if no templates available for this combination
                if not active_tactic_templates and not active_legit_templates:
                    continue
                # 1. Generate Legitimate Messages (~100 per language/script)
                num_legit = 120
                for _ in range(num_legit):
                    if not active_legit_templates:
                        break
                    template = random.choice(active_legit_templates)
                    placeholders = get_placeholders(lang)
                    text = fill_template(template, placeholders)
                    
                    self.dataset.append({
                        "id": str(uuid.uuid4()),
                        "text": text,
                        "language": lang.lower(),
                        "script": "latin" if lang == "English" else ("native" if script == "native" else "romanized"),
                        "label": "legit",
                        "tactics": ""
                    })
                # 2. Generate Single-Tactic Scams (~15 per tactic per language/script)
                for tactic, templates in active_tactic_templates.items():
                    for _ in range(20):
                        template = random.choice(templates)
                        placeholders = get_placeholders(lang)
                        text = fill_template(template, placeholders)
                        # Auto-inject link to verify if tactic requires link but doesn't have it
                        if tactic == "suspicious_link" and "{LINK}" not in template:
                            text += " Click here: " + placeholders["LINK"]
                        self.dataset.append({
                            "id": str(uuid.uuid4()),
                            "text": text,
                            "language": lang.lower(),
                            "script": "latin" if lang == "English" else ("native" if script == "native" else "romanized"),
                            "label": "scam",
                            "tactics": tactic
                        })
                # 3. Generate Multi-Tactic Scams (~60 per language/script)
                # We combine 2 or 3 distinct tactics together
                tactics_list = list(active_tactic_templates.keys())
                for _ in range(60):
                    num_tactics = random.randint(2, 3)
                    selected_tactics = random.sample(tactics_list, num_tactics)
                    
                    parts = []
                    for tactic in selected_tactics:
                        parts.append(random.choice(active_tactic_templates[tactic]))
                    
                    # Merge parts into a single message
                    combined_template = " ".join(parts)
                    placeholders = get_placeholders(lang)
                    text = fill_template(combined_template, placeholders)
                    self.dataset.append({
                        "id": str(uuid.uuid4()),
                        "text": text,
                        "language": lang.lower(),
                        "script": "latin" if lang == "English" else ("native" if script == "native" else "romanized"),
                        "label": "scam",
                        "tactics": ",".join(selected_tactics)
                    })
        # Save to CSV
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text", "language", "script", "label", "tactics"])
            writer.writeheader()
            writer.writerows(self.dataset)
        print(f"Dataset generated successfully! Total samples: {len(self.dataset)}")
if __name__ == "__main__":
    gen = DatasetGenerator()
    gen.generate()
