# Guide: Adding a New Language to ScamShield
ScamShield is designed with an extensible, configuration-first architecture. You can add support for a new language (e.g., Kannada, Malayalam, Marathi, Gujarati, etc.) with minimal coding by defining a new **Language Pack**.
---
## Step 1: Create the Language Pack Configuration
Create a new JSON file under `data/language_packs/` named after your language (e.g., `kannada.json` or `gujarati.json`). 
The file must follow this schema:
```json
{
  "language_name": "Kannada",
  "romanized_words": [
    "maadi", "kodi", "banni", "ila", "illa", "ide", "idhya", "gpay", "phonepe", "paytm"
  ],
  "tactic_templates": {
    "urgency": {
      "native": [
        "ತುರ್ತು! ನಿಮ್ಮ {BANK} ವಿದ್ಯುತ್ ಬಿಲ್ ಅನ್ನು {TIME} ಒಳಗೆ ಪಾವತಿಸಿ ಇಲ್ಲದಿದ್ದರೆ ಇಂದೇ ಸಂಪರ್ಕ ಕಡಿತಗೊಳ್ಳುತ್ತದೆ."
      ],
      "romanized": [
        "Urgent! Mee {BANK} current bill {TIME} kulla pay maadi, illana current cut aaguthe."
      ]
    },
    "authority_impersonation": {
      "native": [
        "RBI ಹೊಸ ನಿಯಮ: ನಿಮ್ಮ ಖಾತೆಯನ್ನು ಸುರಕ್ಷಿತವಾಗಿರಿಸಲು ಈಗಲೇ {LINK} ಕ್ಲಿಕ್ ಮಾಡಿ."
      ],
      "romanized": [
        "RBI hosa rule: Nimma account secure maadalike eega {LINK} click maadi."
      ]
    },
    "false_reward": {
      "native": [
        "ಅಭಿನಂದನೆಗಳು! PhonePe ಮೂಲಕ ನಿಮಗೆ ₹{AMOUNT} ಕ್ಯಾಶ್‌ಬ್ಯಾಕ್ ಸಿಕ್ಕಿದೆ. ಈಗಲೇ ವರ್ಗಾಯಿಸಿ."
      ],
      "romanized": [
        "Congratulations! PhonePe nundi Rs {AMOUNT} cashback bandhidi. Claim maadi: {LINK}"
      ]
    },
    "loss_aversion": {
      "native": [
        "ಎಚ್ಚರಿಕೆ! ನಿಮ್ಮ {BANK} ಖಾತೆಯಲ್ಲಿ ಶಂಕಾಸ್ಪದ ಚಟುವಟಿಕೆ ಕಂಡುಬಂದಿದೆ. ಬ್ಲಾಕ್ ತಪ್ಪಿಸಲು ಕ್ಲಿಕ್ ಮಾಡಿ: {LINK}"
      ],
      "romanized": [
        "Warning! Nimma {BANK} account suspend aagutha ide. Unblock maadalike click maadi: {LINK}"
      ]
    },
    "credential_phishing": {
      "native": [
        "ಖಾತೆ ಸಕ್ರಿಯಗೊಳಿಸಲು ನಿಮ್ಮ ಯುಪಿಐ ಪಿನ್ (UPI PIN) ನಮೂದಿಸಿ: {LINK}"
      ],
      "romanized": [
        "Nimma bank account active maadalike UPI PIN enter maadi: {LINK}"
      ]
    },
    "suspicious_link": {
      "native": [
        "ಈ ಲಿಂಕ್ ಕ್ಲಿಕ್ ಮಾಡಿ: {LINK}"
      ],
      "romanized": [
        "Ee link click maadi: {LINK}"
      ]
    }
  },
  "legit_templates": {
    "native": [
      "ನಿಮ್ಮ {BANK} ಖಾತೆಗೆ ₹{AMOUNT} ಜಮೆಯಾಗಿದೆ. ಮಾಹಿತಿ: XXXX{OTP}."
    ],
    "romanized": [
      "Nimma {BANK} account XXXX{OTP} ge Rs {AMOUNT} credit aagide."
    ]
  }
}
```
### Key Rules for Configs:
- **`language_name`**: The capitalized name of the language (e.g. "Kannada").
- **`romanized_words`**: List of high-frequency transliterated words distinct to that language, used for rule-based romanized text classification.
- **`tactic_templates`**: Sentence templates for the 6 core scam categories, broken into `native` and `romanized` keys. 
- **`legit_templates`**: Safe transaction alerts and OTP notification templates.
---
## Step 2: Register Script Ranges (If using a new Native Script)
If the language uses a script not already supported by the detector, register its Unicode ranges inside [src/detector.py](file:///d:/projects/antigravity%20projects/src/detector.py):
1. **Find the Unicode Block**: Look up the official Unicode block range for the script (e.g., Kannada is `0x0C80` to `0x0CFF`).
2. **Add to `UNICODE_RANGES`**:
   ```python
   UNICODE_RANGES = {
       # ... existing ranges
       "Kannada": (0x0C80, 0x0CFF),
   }
   ```
3. **Add to `SCRIPT_TO_LANG`**:
   ```python
   SCRIPT_TO_LANG = {
       # ... existing mappings
       "Kannada": "Kannada",
   }
   ```
---
## Step 3: Regenerate Dataset and Retrain
Once the language pack is created and registered, run the automation commands from the project root:
1. **Regenerate the CSV Dataset**:
   ```bash
   python scripts/dataset_generator.py
   ```
   The generator will automatically pick up the new JSON pack, load its templates, fill in placeholders, and output an expanded training set to `data/scam_dataset.csv`.
2. **Retrain Classifiers & Re-evaluate**:
   ```bash
   python src/train.py
   ```
   The training pipeline will vectorize the expanded dataset, train new binary and tactic logistic regression heads, and save the updated pickle models under `models/`.
---
## Step 4: Verify the Performance
Open [models/metrics_report.md](file:///d:/projects/antigravity%20projects/models/metrics_report.md) to review the training split validation results. Under the **Performance Breakdown by Language and Script** section, you should see your new language listed along with its native and romanized sub-accuracy scores (Precision, Recall, F1).
