import os
import re
import pickle
import json
import logging
import sqlite3
import uuid
import numpy as np
from typing import Dict, Any, List
from scipy.sparse import hstack
from src.detector import ScriptLanguageDetector
from src.database import log_prediction
from src.model_registry import load_predictor, resolve_model_name
from src.observability import DB_ERRORS, PREDICTIONS, SCORES
from src.pii import scrub_text
from src.taxonomy import TACTICS

log = logging.getLogger("scamshield.inference")

class ScamShieldInference:
    def __init__(
        self, 
        models_dir: str = "models", 
        lang_packs_dir: str = "data/language_packs"
    ):
        self.models_dir = models_dir
        self.lang_packs_dir = lang_packs_dir
        
        # Load detector
        self.detector = ScriptLanguageDetector(lang_packs_dir)
        
        # Load vectorizers and models
        self.word_vec = None
        self.char_vec = None
        self.binary_model = None
        self.tactic_models: Dict[str, Any] = {}
        self.explanations: Dict[str, Any] = {}
        
        # Production predictor (ONNX encoder by default, TF-IDF fallback)
        self.predictor = load_predictor(resolve_model_name())
        
        self.load_artifacts()

    @property
    def model_name(self) -> str:
        return getattr(self.predictor, "name", "unknown")

    def load_artifacts(self):
        # Load vectorizers
        vec_path = os.path.join(self.models_dir, "vectorizers.pkl")
        if os.path.exists(vec_path):
            with open(vec_path, "rb") as f:
                vecs = pickle.load(f)
                self.word_vec = vecs["word"]
                self.char_vec = vecs["char"]
                
        # Load binary classifier
        bin_path = os.path.join(self.models_dir, "binary_classifier.pkl")
        if os.path.exists(bin_path):
            with open(bin_path, "rb") as f:
                self.binary_model = pickle.load(f)
                
        # Load tactic classifiers
        tac_path = os.path.join(self.models_dir, "tactic_classifier.pkl")
        if os.path.exists(tac_path):
            with open(tac_path, "rb") as f:
                self.tactic_models = pickle.load(f)
        # Load explanations config (English as default)
        exp_path = os.path.join(self.lang_packs_dir, "explanations_en.json")
        if os.path.exists(exp_path):
            with open(exp_path, "r", encoding="utf-8") as f:
                self.explanations = json.load(f)
    def extract_evidence(self, text: str, tactic: str) -> str:
        """
        Identifies the exact phrase or word in the text that triggered a tactic
        by combining TF-IDF feature weights and model coefficients.
        """
        if not self.word_vec or not self.char_vec or tactic not in self.tactic_models:
            return "suspicious phrasing"
        model = self.tactic_models[tactic]
        coefs = model.coef_[0]
        # Get features present in the input text
        X_word = self.word_vec.transform([text])
        X_char = self.char_vec.transform([text])
        word_indices = X_word.nonzero()[1]
        char_indices = X_char.nonzero()[1]
        word_names = self.word_vec.get_feature_names_out()
        char_names = self.char_vec.get_feature_names_out()
        word_len = len(word_names)
        candidates = []
        # Calculate influence score = TF-IDF value * Logistic Regression coefficient
        for idx in word_indices:
            term = word_names[idx]
            score = X_word[0, idx] * coefs[idx]
            if score > 0: # Only positive contributors
                candidates.append((term, score))
        for idx in char_indices:
            term = char_names[idx]
            score = X_char[0, idx] * coefs[word_len + idx]
            if score > 0:
                candidates.append((term, score))
        # Sort candidates by influence score
        candidates.sort(key=lambda x: x[1], reverse=True)
        # Retrieve the best term
        if candidates:
            best_term, _ = candidates[0]
            
            # If the feature is a character n-gram, find the full word containing it
            words_in_text = re.findall(r'\b\w+\b', text.lower())
            if len(best_term) <= 5 and best_term not in words_in_text:
                for w in words_in_text:
                    if best_term in w and len(w) > len(best_term):
                        return w
            return best_term
        return "suspicious phrasing"
    def build_explanation(self, detected_tactics_list: List[Dict[str, Any]], label: str = "scam") -> str:
        """
        Constructs a plain-language explanation of why the message was flagged.

        ``label`` is passed explicitly because a model can flag a message as a scam
        without clearing the threshold on any single tactic head; keying the
        explanation off the tactic list alone would then describe a scam as
        legitimate.
        """
        templates = self.explanations.get("explanation_templates", {})
        if not self.explanations:
            if label == "legit":
                return "This message appears to be legitimate. Keep in mind to never share your credentials."
            return "This message looks like a scam. Do not click links or share OTPs or PINs."
        
        if label == "legit":
            return templates.get(
                "legit_explanation", 
                "This message appears to be legitimate. Keep in mind to never share your credentials."
            )
        if not detected_tactics_list:
            return (
                "This message was classified as a scam, but no single manipulation tactic "
                "passed its threshold. Treat it as suspicious: do not click links or share "
                "an OTP, UPI PIN or password."
            )
        header = templates.get("scam_header", "This message exhibits scam tactics:").replace("{COUNT}", str(len(detected_tactics_list)))
        bullets = []
        tactic_names = self.explanations.get("tactic_names", {})
        tactic_descs = self.explanations.get("tactic_descriptions", {})
        bullet_tmpl = templates.get("scam_tactic_bullet", "- **{TACTIC_NAME}**: {TACTIC_DESC} (Triggered by: \"{EVIDENCE}\")")
        for dt in detected_tactics_list:
            t_id = dt["tactic"]
            t_name = tactic_names.get(t_id, t_id.replace("_", " ").title())
            t_desc = tactic_descs.get(t_id, "Manipulation pattern detected.")
            evidence = dt["evidence"]
            
            bullet = bullet_tmpl.replace("{TACTIC_NAME}", t_name).replace("{TACTIC_DESC}", t_desc).replace("{EVIDENCE}", evidence)
            bullets.append(bullet)
        footer = templates.get("scam_footer", "")
        
        return f"{header}\n" + "\n".join(bullets) + (f"\n\n{footer}" if footer else "")
    def analyze(self, text: str, source: str = "web") -> Dict[str, Any]:
        """
        Analyzes the input message, returning script type, language, scam probability,
        tactic classification, evidence extraction, and summary explanation.
        """
        message_id = str(uuid.uuid4())
        
        # 1. Run Detector
        detection = self.detector.detect(text)
        language_guess = detection["language_guess"]
        script_type = detection["script_type"]
        detected_script = detection["detected_script"]
        
        # 2. Score with the production predictor (ONNX encoder or TF-IDF baseline)
        prediction = self.predictor.predict([text])[0]
        scam_prob = float(prediction.scam_probability)
        label = "scam" if scam_prob >= 0.5 else "legit"
        
        # 3. Tactics from the same model, with verbatim evidence spans
        detected_tactics = []
        if label == "scam":
            for tactic_prediction in prediction.tactics:
                evidence = self.evidence_for(text, tactic_prediction.tactic)
                detected_tactics.append({
                    "tactic": tactic_prediction.tactic,
                    "confidence": round(tactic_prediction.probability, 2),
                    "evidence": evidence
                })
        
        # 4. Build Explanation
        explanation = self.build_explanation(detected_tactics, label)
        SCORES.labels(self.model_name).observe(scam_prob)
        PREDICTIONS.labels(label, language_guess.lower(), script_type.lower()).inc()
        log.info(
            "prediction",
            extra={"label": label, "score": round(scam_prob, 4), "language": language_guess,
                   "script": script_type, "tactics": len(detected_tactics), "source": source},
        )
        # The analysis log is for dashboards and drift checks, not for serving: if
        # the database is unavailable the user still gets an answer.  Stored text is
        # scrubbed because the live service receives real people's messages.
        try:
            log_prediction(
                message_id, scrub_text(text), label, scam_prob,
                [dt["tactic"] for dt in detected_tactics], language_guess, script_type, source,
            )
        except sqlite3.Error:
            DB_ERRORS.labels("log_prediction").inc()
            log.exception("could not write the analysis log")
        return {
            "message_id": message_id,
            "scam_probability": round(scam_prob, 2),
            "label": label,
            "language_detected": language_guess,
            "script_type": script_type,
            "detected_script": detected_script,
            "tactics": detected_tactics,
            "explanation": explanation,
            "model": self.model_name
        }
    
    def evidence_for(self, text: str, tactic: str) -> str:
        """A verbatim span of the message that supports the tactic, or generic phrasing.
        
        The TF-IDF baseline can attribute a span from its coefficients. The neural
        predictor cannot (it has no per-token weights), so it returns generic phrasing
        rather than inventing a quote - see docs/robustness.md on why quoting text the
        sender did not write is unacceptable.
        """
        if hasattr(self.predictor, "evidence"):
            return self.predictor.evidence(text, tactic)
        return "suspicious phrasing"
if __name__ == "__main__":
    analyzer = ScamShieldInference()
    
    # Test scam
    scam_text = "Urgent! Pay your electricity bill within 10 minutes or SBI will disconnect your connection tonight. Click to pay: bit.ly/pay-upi-ref"
    res = analyzer.analyze(scam_text)
    print(json.dumps(res, indent=2))
