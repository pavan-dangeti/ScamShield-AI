import os
import sys
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
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
# Load environment configurations
load_dotenv()
from src.inference import ScamShieldInference
from src.database import init_db, save_feedback, get_stats, update_log_source
# Initialize Database on startup
init_db()
# Twilio signature validation credentials.
# Validation is ON by default. Set SKIP_TWILIO_VALIDATION=true only for local
# testing without ngrok; a deployed webhook that skips validation lets anyone
# POST forged messages and read the analysis response.
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
SKIP_TWILIO_VALIDATION = os.getenv("SKIP_TWILIO_VALIDATION", "false").lower() == "true"
if SKIP_TWILIO_VALIDATION and not os.getenv("RENDER"):
    print("WARNING: Twilio signature validation is disabled.")

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


def rate_limit(request: Request) -> None:
    """Reject clients that exceed RATE_LIMIT_PER_MINUTE requests per window."""
    client = request.client.host if request.client else "unknown"
    now = time.time()
    hits = [stamp for stamp in _rate_state.get(client, []) if now - stamp < RATE_LIMIT_WINDOW]
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests; slow down.")
    hits.append(now)
    _rate_state[client] = hits

# Initialize Inference Engine
analyzer = ScamShieldInference()
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
# Request Models
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Suspected scam message text", min_length=2)
class FeedbackRequest(BaseModel):
    message_id: str = Field(..., description="UUID of analyzed message")
    user_correction: str = Field(..., description="Correction value: 'scam' or 'legit'")
@app.post("/api/analyze", dependencies=[Depends(rate_limit)])
async def analyze_message_endpoint(req: AnalyzeRequest):
    try:
        result = analyzer.analyze(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
@app.post("/api/sms-webhook")
async def twilio_sms_webhook(request: Request):
    """
    Receives incoming text messages from Twilio via url-encoded POST.
    Performs signature verification, runs inference, and returns TwiML reply.
    """
    form_data = await request.form()
    body = form_data.get("Body", "").strip()
    sender = form_data.get("From", "").strip()
    
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
        # Run inference
        result = analyzer.analyze(body)
        
        # Set source to 'sms' in database
        update_log_source(result["message_id"], "sms")
        
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
        
    except Exception as e:
        # Prevent throwing 500 error to Twilio, return a valid TwiML with fallback error message
        response = MessagingResponse()
        response.message("⚠️ ScamShield is temporarily unable to analyze this message.")
        return Response(content=str(response), media_type="application/xml")
@app.post("/api/feedback")
async def save_feedback_endpoint(req: FeedbackRequest):
    correction = req.user_correction.strip().lower()
    if correction not in ["scam", "legit"]:
        raise HTTPException(status_code=400, detail="Correction must be 'scam' or 'legit'")
    try:
        save_feedback(req.message_id, correction)
        return {"status": "success", "message": "Feedback recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
@app.get("/api/health")
async def health():
    """Liveness plus which model is actually serving, so a broken deploy is visible."""
    return {
        "status": "ok",
        "model": getattr(analyzer.predictor, "name", "unknown"),
        "trained": True,
    }


@app.get("/api/stats")
async def get_stats_endpoint():
    try:
        stats = get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database statistics error: {str(e)}")
# Mount static files for the web interface
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
