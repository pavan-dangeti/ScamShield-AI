import logging
import os
import sys
import threading
import time
from typing import Dict
# Add local target library directory to sys.path to resolve packages on Windows
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")))
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
# Load environment configurations
load_dotenv()
from src.observability import MODEL_INFO, RequestContextMiddleware, configure_logging, request_id_var

configure_logging(os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("scamshield.api")

from src.inference import ScamShieldInference
from src.database import init_db, save_feedback, get_stats
# Initialize Database on startup
init_db()
# Twilio signature validation credentials.
# Validation is ON by default. Set SKIP_TWILIO_VALIDATION=true only for local
# testing without ngrok; a deployed webhook that skips validation lets anyone
# POST forged messages and read the analysis response.
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
SKIP_TWILIO_VALIDATION = os.getenv("SKIP_TWILIO_VALIDATION", "false").lower() == "true"
if SKIP_TWILIO_VALIDATION:
    log.warning("Twilio signature validation is disabled")

# CORS: the dashboard is served by this app, so third-party origins are not needed.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",")
    if origin.strip()
]
# Simple in-process rate limit for the public analysis endpoint.
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
_rate_state: Dict[str, list[float]] = {}
RATE_LIMIT_WINDOW = 60.0
# Sync endpoints and dependencies run in a threadpool, so the shared state needs a lock.
_rate_lock = threading.Lock()
# Longest message accepted.  Real SMS/WhatsApp scams are far shorter; the cap stops
# one request from tying up a worker with megabytes of text.
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "2000"))


def rate_limit(request: Request) -> None:
    """Reject clients that exceed RATE_LIMIT_PER_MINUTE requests per window."""
    client = request.client.host if request.client else "unknown"
    now = time.time()
    with _rate_lock:
        hits = [stamp for stamp in _rate_state.get(client, []) if now - stamp < RATE_LIMIT_WINDOW]
        if len(hits) >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Too many requests; slow down.")
        hits.append(now)
        _rate_state[client] = hits


def internal_error(what: str) -> HTTPException:
    """Log the real error with its request ID; tell the client only how to report it."""
    log.exception(what)
    return HTTPException(status_code=500, detail=f"{what}. Quote request ID {request_id_var.get()} when reporting it.")

# Initialize Inference Engine
analyzer = ScamShieldInference()
MODEL_INFO.labels(analyzer.model_name).set(1)
log.info("model loaded", extra={"model": analyzer.model_name})
app = FastAPI(
    title="ScamShield API",
    description="Regional-Language UPI/Payment Scam Detector with Tactic Explainer",
    version="1.0.0"
)
# CORS restricted to configured origins (the dashboard is same-origin).
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.add_middleware(RequestContextMiddleware)
# Request Models
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Suspected scam message text", min_length=2, max_length=MAX_MESSAGE_CHARS)
class FeedbackRequest(BaseModel):
    message_id: str = Field(..., description="UUID of analyzed message")
    user_correction: str = Field(..., description="Correction value: 'scam' or 'legit'")
@app.post("/api/analyze", dependencies=[Depends(rate_limit)])
def analyze_message_endpoint(req: AnalyzeRequest):
    try:
        return analyzer.analyze(req.text)
    except Exception:
        raise internal_error("Inference failed")
@app.post("/api/sms-webhook")
async def twilio_sms_webhook(request: Request):
    """
    Receives incoming text messages from Twilio via url-encoded POST.
    Performs signature verification, runs inference, and returns TwiML reply.
    """
    form_data = await request.form()
    body = str(form_data.get("Body", "")).strip()

    if not body:
        raise HTTPException(status_code=400, detail="Missing Body parameter")
        
    # Signature Verification (Skip on local testing)
    if not SKIP_TWILIO_VALIDATION:
        signature = request.headers.get("x-twilio-signature")
        if not signature:
            raise HTTPException(status_code=403, detail="Missing X-Twilio-Signature header")
            
        # Proxy URL Reconstruction for ngrok compatibility
        forwarded_proto = request.headers.get("x-forwarded-proto", "http")
        forwarded_host = request.headers.get("x-forwarded-host", request.url.netloc)
        url = f"{forwarded_proto}://{forwarded_host}{request.url.path}"
        
        # Convert form attributes to dict
        params = {k: v for k, v in form_data.items()}
        
        validator = RequestValidator(TWILIO_AUTH_TOKEN)
        if not validator.validate(url, params, signature):
            raise HTTPException(status_code=403, detail="Invalid Twilio request signature")
            
    try:
        result = await run_in_threadpool(analyzer.analyze, body[:MAX_MESSAGE_CHARS], "sms")
        
        # Construct reply message (strictly capped at 320 characters)
        prob = int(result["scam_probability"] * 100)
        
        if result["label"] == "scam":
            tactic_names = [t["tactic"].replace("_", " ").title() for t in result["tactics"]]
            tactic_str = ", ".join(tactic_names[:3])
            reply_text = f"⚠️ This message looks like a SCAM ({prob}% confidence)."
            if tactic_str:
                reply_text += f" Detected: {tactic_str}."
            reply_text += " Do not click links or share OTP/PIN."
        else:
            reply_text = f"✅ This message looks safe ({prob}% scam confidence)."
            
        # Guard rails for length limit (2 SMS segments maximum)
        if len(reply_text) > 320:
            reply_text = reply_text[:317] + "..."
            
        # Build TwiML MessagingResponse XML
        response = MessagingResponse()
        response.message(reply_text)
        
        return Response(content=str(response), media_type="application/xml")
        
    except Exception:
        # Twilio retries on a 500; a valid TwiML apology is the better failure mode.
        log.exception("webhook analysis failed")
        response = MessagingResponse()
        response.message("⚠️ ScamShield is temporarily unable to analyze this message.")
        return Response(content=str(response), media_type="application/xml")
@app.post("/api/feedback")
def save_feedback_endpoint(req: FeedbackRequest):
    correction = req.user_correction.strip().lower()
    if correction not in ["scam", "legit"]:
        raise HTTPException(status_code=400, detail="Correction must be 'scam' or 'legit'")
    try:
        updated = save_feedback(req.message_id, correction)
    except Exception:
        raise internal_error("Could not save feedback")
    if not updated:
        raise HTTPException(status_code=404, detail="Unknown message_id")
    return {"status": "success", "message": "Feedback recorded successfully."}
@app.get("/api/health")
def health():
    """Liveness plus which model is actually serving, so a silent fallback is visible."""
    return {"status": "ok", "model": analyzer.model_name}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/stats")
def get_stats_endpoint():
    try:
        return get_stats()
    except Exception:
        raise internal_error("Could not read statistics")
# Mount static files for the web interface
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
