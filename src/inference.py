import json
import logging
import os
import sqlite3
import uuid
from typing import Any, Dict, List

from src.database import log_prediction
from src.detector import ScriptLanguageDetector
from src.model_registry import load_predictor, resolve_model_name
from src.observability import DB_ERRORS, PREDICTIONS, SCORES
from src.pii import scrub_text

log = logging.getLogger("scamshield.inference")


class ScamShieldInference:
    def __init__(self, lang_packs_dir: str = "data/language_packs"):
        self.detector = ScriptLanguageDetector(lang_packs_dir)
        self.predictor = load_predictor(resolve_model_name())
        self.explanations: Dict[str, Any] = {}
        exp_path = os.path.join(lang_packs_dir, "explanations_en.json")
        if os.path.exists(exp_path):
            with open(exp_path, encoding="utf-8") as handle:
                self.explanations = json.load(handle)

    @property
    def model_name(self) -> str:
        return getattr(self.predictor, "name", "unknown")

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
        header = templates.get("scam_header", "This message exhibits scam tactics:").replace(
            "{COUNT}", str(len(detected_tactics_list))
        )
        tactic_names = self.explanations.get("tactic_names", {})
        tactic_descs = self.explanations.get("tactic_descriptions", {})
        bullet_tmpl = templates.get(
            "scam_tactic_bullet", "- **{TACTIC_NAME}**: {TACTIC_DESC} (Triggered by: \"{EVIDENCE}\")"
        )
        bullets = []
        for dt in detected_tactics_list:
            t_id = dt["tactic"]
            t_name = tactic_names.get(t_id, t_id.replace("_", " ").title())
            t_desc = tactic_descs.get(t_id, "Manipulation pattern detected.")
            bullets.append(
                bullet_tmpl.replace("{TACTIC_NAME}", t_name)
                .replace("{TACTIC_DESC}", t_desc)
                .replace("{EVIDENCE}", dt["evidence"])
            )
        footer = templates.get("scam_footer", "")
        return f"{header}\n" + "\n".join(bullets) + (f"\n\n{footer}" if footer else "")

    def analyze(self, text: str, source: str = "web") -> Dict[str, Any]:
        """Language and script, scam probability, tactics with evidence, and an explanation."""
        message_id = str(uuid.uuid4())
        detection = self.detector.detect(text)
        language_guess = detection["language_guess"]
        script_type = detection["script_type"]

        prediction = self.predictor.predict([text])[0]
        scam_prob = float(prediction.scam_probability)
        label = "scam" if scam_prob >= 0.5 else "legit"

        detected_tactics: List[Dict[str, Any]] = []
        if label == "scam":
            for tactic_prediction in prediction.tactics:
                detected_tactics.append({
                    "tactic": tactic_prediction.tactic,
                    "confidence": round(tactic_prediction.probability, 2),
                    "evidence": self.evidence_for(text, tactic_prediction.tactic),
                })
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
            "detected_script": detection["detected_script"],
            "tactics": detected_tactics,
            "explanation": explanation,
            "model": self.model_name,
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
