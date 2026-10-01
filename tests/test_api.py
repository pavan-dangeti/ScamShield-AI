"""API behaviour, including what happens when parts of the system fail.

Each failure test breaks one dependency on purpose (the model, the database, the
webhook signature, the client's request rate, the input size) and checks that
the service degrades the way the design says it should, without leaking internals.
"""

from __future__ import annotations

import importlib
import logging
import sqlite3
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from twilio.request_validator import RequestValidator

from src import database, inference, model_registry
from src.models.tfidf import TfidfScamModel

SCAM = "Urgent: your account will be blocked in 5 minutes, click bit.ly/abc9 now"
WEBHOOK = "http://testserver/api/sms-webhook"
TOKEN = "test-auth-token"


class Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def api(tmp_path, monkeypatch, tiny_model):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "feedback.db"))
    monkeypatch.setattr(inference, "load_predictor", lambda name: tiny_model)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("SKIP_TWILIO_VALIDATION", "false")
    import src.main

    main = importlib.reload(src.main)  # re-read the environment set above
    records = Records()
    logging.getLogger().addHandler(records)
    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client, main, records
    logging.getLogger().removeHandler(records)


def metric(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def signed(params: dict[str, str]) -> dict[str, str]:
    return {"X-Twilio-Signature": RequestValidator(TOKEN).compute_signature(WEBHOOK, params)}


def stored(column: str) -> str:
    conn = sqlite3.connect(database.DB_PATH)
    (value,) = conn.execute(f"SELECT {column} FROM analysis_logs").fetchone()
    conn.close()
    return value


# --- normal operation ----------------------------------------------------------


def test_analyze_returns_a_prediction_and_a_request_id(api):
    client, _, _ = api
    response = client.post("/api/analyze", json={"text": SCAM})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("scam", "legit") and 0 <= body["scam_probability"] <= 1
    assert body["model"] == "tfidf-lr"
    assert len(response.headers["x-request-id"]) == 32


def test_a_safe_caller_request_id_is_kept_and_an_unsafe_one_replaced(api):
    client, _, _ = api
    assert client.get("/api/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"
    injected = client.get("/api/health", headers={"X-Request-ID": 'x" level="CRITICAL'})
    assert injected.headers["x-request-id"] != 'x" level="CRITICAL'


def test_health_reports_the_model_actually_serving(api):
    client, _, _ = api
    assert client.get("/api/health").json() == {"status": "ok", "model": "tfidf-lr"}


def test_metrics_use_route_templates_and_record_scores(api):
    client, _, _ = api
    labels = {"route": "/api/analyze", "method": "POST", "status": "200"}
    before = metric("scamshield_http_requests_total", **labels)
    scores = metric("scamshield_scam_probability_count", model="tfidf-lr")
    client.post("/api/analyze", json={"text": SCAM})
    assert metric("scamshield_http_requests_total", **labels) == before + 1
    assert metric("scamshield_scam_probability_count", model="tfidf-lr") == scores + 1
    assert "scamshield_http_request_duration_seconds_bucket" in client.get("/metrics").text


def test_feedback_updates_a_known_message_and_rejects_an_unknown_one(api):
    client, _, _ = api
    message_id = client.post("/api/analyze", json={"text": SCAM}).json()["message_id"]
    assert client.post("/api/feedback", json={"message_id": message_id, "user_correction": "legit"}).status_code == 200
    assert client.post("/api/feedback", json={"message_id": "nope", "user_correction": "legit"}).status_code == 404
    assert client.post("/api/feedback", json={"message_id": message_id, "user_correction": "maybe"}).status_code == 400


# --- privacy -------------------------------------------------------------------


def test_stored_messages_are_scrubbed_of_personal_data(api):
    client, _, _ = api
    client.post("/api/analyze", json={"text": "Call +91 98765 43210 or mail ravi.k@example.com to claim"})
    text = stored("text")
    assert "98765" not in text and "ravi.k@example.com" not in text
    assert "[PHONE]" in text and "[EMAIL]" in text


def test_logs_never_contain_the_message_text(api):
    client, _, records = api
    client.post("/api/analyze", json={"text": "canary-7f3e2 your account will be blocked"})
    assert records.records, "expected request and prediction log lines"
    assert not any("canary-7f3e2" in str(record.__dict__) for record in records.records)


# --- failures, deliberately induced --------------------------------------------


def test_input_that_is_too_long_empty_or_malformed_is_rejected(api):
    client, main, _ = api
    assert client.post("/api/analyze", json={"text": "x" * (main.MAX_MESSAGE_CHARS + 1)}).status_code == 422
    assert client.post("/api/analyze", json={"text": ""}).status_code == 422
    malformed = client.post("/api/analyze", content=b"{not json", headers={"Content-Type": "application/json"})
    assert malformed.status_code == 422


def test_model_failure_returns_a_generic_500_and_logs_the_cause(api, monkeypatch):
    client, main, records = api

    def broken(texts):
        raise RuntimeError("tensor shape mismatch in layer 7")

    monkeypatch.setattr(main.analyzer.predictor, "predict", broken)
    response = client.post("/api/analyze", json={"text": SCAM})
    assert response.status_code == 500
    assert "tensor shape" not in response.text, "internal error detail leaked to the client"
    assert response.headers["x-request-id"] in response.json()["detail"]
    logged = [r for r in records.records if r.exc_info and "tensor shape" in str(r.exc_info[1])]
    assert logged and logged[0].levelno == logging.ERROR


def test_database_outage_does_not_stop_analysis(api, monkeypatch):
    client, _, _ = api

    def down(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(inference, "log_prediction", down)
    before = metric("scamshield_db_errors_total", operation="log_prediction")
    response = client.post("/api/analyze", json={"text": SCAM})
    assert response.status_code == 200 and response.json()["label"] in ("scam", "legit")
    assert metric("scamshield_db_errors_total", operation="log_prediction") == before + 1


def test_stats_failure_is_a_generic_500(api, monkeypatch):
    client, main, _ = api

    def down():
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(main, "get_stats", down)
    response = client.get("/api/stats")
    assert response.status_code == 500 and "not a database" not in response.text


def test_rate_limit_answers_429(api, monkeypatch):
    client, main, _ = api
    monkeypatch.setattr(main, "RATE_LIMIT", 2)
    main._rate_state.clear()
    codes = [client.post("/api/analyze", json={"text": SCAM}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_webhook_rejects_missing_and_forged_signatures(api):
    client, _, _ = api
    params = {"Body": SCAM, "From": "+15550001111"}
    assert client.post("/api/sms-webhook", data=params).status_code == 403
    forged = {"X-Twilio-Signature": "bm90IGEgcmVhbCBzaWduYXR1cmU="}
    assert client.post("/api/sms-webhook", data=params, headers=forged).status_code == 403


def test_webhook_replies_with_twiml_and_logs_the_source(api):
    client, _, _ = api
    params = {"Body": SCAM, "From": "+15550001111"}
    response = client.post("/api/sms-webhook", data=params, headers=signed(params))
    assert response.status_code == 200 and "xml" in response.headers["content-type"]
    reply = ET.fromstring(response.text).findtext("Message") or ""
    assert "SCAM" in reply or "safe" in reply
    assert stored("source") == "sms"


def test_webhook_model_failure_still_answers_twilio_with_valid_twiml(api, monkeypatch):
    client, main, _ = api

    def broken(texts):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(main.analyzer.predictor, "predict", broken)
    params = {"Body": SCAM, "From": "+15550001111"}
    response = client.post("/api/sms-webhook", data=params, headers=signed(params))
    assert response.status_code == 200, "a 5xx makes Twilio retry the same message"
    assert "unable to analyze" in (ET.fromstring(response.text).findtext("Message") or "")


def test_missing_or_corrupt_model_artefact_falls_back_to_the_baseline(tmp_path, monkeypatch, tiny_model):
    monkeypatch.setattr(TfidfScamModel, "load", staticmethod(lambda path: tiny_model))
    monkeypatch.setattr(model_registry, "onnx_path_for", lambda name: str(tmp_path / "missing.onnx"))
    assert model_registry.load_predictor("xlmr-aug-int8") is tiny_model

    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"this is not an onnx graph")
    monkeypatch.setattr(model_registry, "onnx_path_for", lambda name: str(corrupt))
    assert model_registry.load_predictor("xlmr-aug-int8") is tiny_model
