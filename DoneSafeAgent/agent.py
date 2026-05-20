"""
Safety Records Agent (DoneSafe) Validation Harness.
Implements a concurrent background HTTP daemon thread serving index.html (fallback),
hosts the core ADK agent loop, and provides native web-page rendering
delivery alongside direct A2A peer task delegation routines.
"""

import os
import json
import logging
import threading
import http.server
import urllib.request
import time
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types

load_dotenv()

# Setup persistence logging matching your organizational pattern
logging.basicConfig(
    level=logging.INFO, 
    filename='a2ui_agent_submissions.log',
    format='%(asctime)s [%(levelname)s] (DoneSafe Engine): %(message)s'
)
logger = logging.getLogger(__name__)

# Framework constant path layout matching your production profile environment
FLASH_PATH = "gemini-2.5-flash"

GEMINI_RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=6,
    initial_delay=1.0,
    exp_base=2.0,
)

# Filepath for persistent form state exchange between backend thread and tool execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_FILE = os.path.join(BASE_DIR, "latest_submission.json")
SCHEMA_FILE = os.path.join(BASE_DIR, "ui_schema.json")

# -----------------------------------------------------------------------------
# ✨ Gemini API Direct Integration Handler (Exponential Backoff Implementation)
# -----------------------------------------------------------------------------

def call_gemini_api(prompt: str, system_instruction: str = "") -> str:
    """
    Directly queries Google's Gemini LLM service with an exponential backoff retry loop.
    Strictly uses the 'gemini-2.5-flash-preview-09-2025' model as required.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
        
    headers = {
        "Content-Type": "application/json"
    }
    
    # Exponential backoff retry loop up to 5 times with delays of 1s, 2s, 4s, 8s, 16s
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode('utf-8'), 
                headers=headers, 
                method='POST'
            )
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                text = res_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return text.strip()
        except Exception as e:
            # Silence retries to comply with console logging guidelines, delay next turn
            time.sleep(2 ** attempt)
            
    return "Error: Unable to safely construct AI safety assessment. Verify your Gemini API credentials."

# -----------------------------------------------------------------------------
# Background Companion Server hosting local assets and post endpoints
# -----------------------------------------------------------------------------
COMPANION_PORT = 8089  # Static port for companion web host

class CompanionHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    HTTP handler serving index.html / form_preview.html and appending post submissions
    directly into the logging pipeline and writing to a persistent JSON file.
    """
    def log_message(self, format, *args):
        # Mute logging normal console hits to prevent trace panel loops
        pass

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        self.end_headers()

    def do_GET(self):
        # Fallback path checking for local development convenience
        file_path = os.path.join(BASE_DIR, "index.html")
        if not os.path.exists(file_path):
            file_path = os.path.join(BASE_DIR, "form_preview.html")

        if os.path.exists(file_path):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Interactive form asset template missing from directory.")

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        self.end_headers()

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            payload = json.loads(post_data.decode('utf-8'))
        except Exception as e:
            self.wfile.write(json.dumps({"status": "ERROR", "message": f"Malformed payload JSON: {e}"}).encode('utf-8'))
            return

        if self.path == '/submit':
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            
            # Write directly to physical file cache to persist state across turn-based process reloads
            try:
                with open(SUBMISSION_FILE, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to write physical state cache: {e}")

            log_entry = {
                "agent": "Safety Records Agent (DoneSafe)",
                "timestamp": timestamp,
                "record_payload": payload
            }
            logger.info("A2UI DONE-SAFE RECORD INGESTION: %s", json.dumps(log_entry))
            response = {"status": "OK", "message": "Record cataloged successfully"}
            self.wfile.write(json.dumps(response).encode('utf-8'))

        elif self.path == '/api/generate_snapshot':
            name = payload.get("name", "Unnamed worker")
            job_type = payload.get("type", "Employee")
            dept = payload.get("department", "General store operations")
            skills = ", ".join(payload.get("specialised_skills", [])) or "Standard compliance skills"
            locations = ", ".join(payload.get("location", [])) or "Assigned shop floor"
            
            prompt = (
                f"Draft a professional retail worker safety snapshot profile paragraph "
                f"for {name}, working as an assigned {job_type} in the {dept} department. "
                f"Their specialized skill matrix includes: {skills}. "
                f"Their assigned store regional locations are: {locations}. "
                f"Incorporate their designated skills and locations into a safety-compliant, positive narrative. "
                f"Keep the summary strictly to 3 concise sentences. Write in the third person. "
                f"Do not include placeholders, return raw plain text with no quotes."
            )
            system_instruction = "You are a senior safety officer at Woolworths. You compose highly polished, compact employee safety snapshots."
            
            result = call_gemini_api(prompt, system_instruction)
            response = {"status": "OK", "result": result}
            self.wfile.write(json.dumps(response).encode('utf-8'))

        elif self.path == '/api/generate_advice':
            action_type = payload.get("action_type", "guidelines")
            name = payload.get("name", "the worker")
            dept = payload.get("department", "general storefront operations")
            skills = ", ".join(payload.get("specialised_skills", [])) or "basic safety protocols"
            locations = ", ".join(payload.get("location", [])) or "the general retail floor"
            
            if action_type == "guidelines":
                prompt = (
                    f"Create an operational SWMS Safe Work Method Statement for {name} based on their assigned department: {dept}. "
                    f"Their specialised skills are: {skills}. They are working in areas: {locations}. "
                    f"Draft exactly 3 high-impact, actionable safety precautions tailored specifically to these skills. "
                    f"Format with clean numbers, clear bold highlights, and keep it very brief."
                )
            else:
                prompt = (
                    f"Generate a realistic, Woolworths-specific hazard incident roleplay question card involving the {dept} department. "
                    f"The scenario must explicitly relate to handling operations using the skills: {skills}. "
                    f"After the 3-sentence hazard scenario, write exactly one safety review response question to test if {name} "
                    f"knows how to react according to standard protocols."
                )
                
            system_instruction = "You are a professional retail risk compliance advisor for Woolworths, writing custom interactive training modules."
            
            result = call_gemini_api(prompt, system_instruction)
            response = {"status": "OK", "result": result}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


class ThreadingHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True


def start_companion_server():
    try:
        server_address = ('', COMPANION_PORT)
        httpd = ThreadingHTTPServer(server_address, CompanionHTTPRequestHandler)
        logger.info(f"[!] Companion background server listening on port {COMPANION_PORT}")
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"Failed to boot daemon companion server: {e}")


server_thread = threading.Thread(target=start_companion_server, daemon=True)
server_thread.start()

# -----------------------------------------------------------------------------
# Core Model Builder
# -----------------------------------------------------------------------------
def build_gemini_model(
    model_path: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    output_tokens: Optional[int] = None,
    thinking_level: Optional[str] = None,
    tools: Optional[list] = None,
    disable_safety: bool = False,
) -> Gemini:
    config_args = {}

    if temperature is not None:
        config_args["temperature"] = temperature
    if top_p is not None:
        config_args["top_p"] = top_p
    if output_tokens is not None:
        config_args["max_output_tokens"] = output_tokens
    if thinking_level is not None:
        config_args["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level
        )
    if tools is not None:
        config_args["tools"] = tools

    if disable_safety:
        config_args["safety_settings"] = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]

    logger.info(
        "[MODEL INIT] model=%s temp=%s top_p=%s max_tokens=%s thinking=%s tools=%s",
        model_path, temperature, top_p, output_tokens, thinking_level, bool(tools),
    )
    
    return Gemini(
        model=model_path,
        retry_options=GEMINI_RETRY_OPTIONS,
        generate_content_config=types.GenerateContentConfig(**config_args) if config_args else None,
    )

# -----------------------------------------------------------------------------
# Web Page Rendering Function Tools
# -----------------------------------------------------------------------------

def render_safety_form() -> str:
    """
    Returns your public secure GitHub Pages URL.
    This secure URL will be parsed by the Agentspace framework to render your
    interactive form inside the right-hand Canvas panel!
    """
    return "https://Ruchi-K.github.io/SHW-agent/"


def verify_latest_submission() -> str:
    """
    Reads the cached form state submitted by the user, and initiates
    an autonomous A2A delegation review with the Store Compliance Auditor Agent.
    """
    if not os.path.exists(SUBMISSION_FILE):
        return (
            "Verification Failed: No recent form submissions detected. "
            "Please open the form, populate the parameters, and click "
            "'Submit & Record Data' before asking me to verify."
        )

    try:
        with open(SUBMISSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        os.remove(SUBMISSION_FILE)
    except Exception as e:
        return f"Error reading state file: {e}"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # -------------------------------------------------------------------------
    # Autonomous A2A Task Delegation Loop
    # -------------------------------------------------------------------------
    dispatch_payload = {
        "task_id": f"AUDIT-{data.get('job_id', '99999')}",
        "worker_name": data.get('name', 'N/A'),
        "department": data.get('department', 'None Selected'),
        "specialised_skills": data.get('specialised_skills', []),
        "assigned_locations": data.get('location', [])
    }
    
    # Simulate routing task request over JSON interface to Compliance Auditor Agent
    auditor_prompt = (
        f"You have received a safety audit delegation payload from safety_records_agent_donesafe:\n"
        f"{json.dumps(dispatch_payload)}\n\n"
        f"Analyze this payload. Check if their specialized skills match their assigned department.\n"
        f"Particularly: If they are in the 'eCommerce' or 'Safety' department, they MUST have "
        f"'Safe Manual Handling' or 'Food Safety Compliance' depending on role contexts.\n"
        f"Generate a professional, structured A2A compliance response audit letter.\n"
        f"Include a 'Audit Status' (APPROVED or PROVISIONAL CLEARANCE with conditions).\n"
        f"Keep the audit letter to exactly 4 sentences. Address the DoneSafe Agent directly. Do not include placeholders."
    )
    
    auditor_system_prompt = (
        "You are the specialized autonomous 'Store Compliance Auditor Agent' at Woolworths. "
        "You process JSON audit requests dispatched to you by peer coordinator agents and return strict regulatory reviews."
    )
    
    # Query simulated auditor peer agent
    auditor_agent_response = call_gemini_api(auditor_prompt, auditor_system_prompt)

    return f"""
### [✓] DoneSafe Safety Record Captured Natively!
**Timestamp Captured:** `{timestamp}`  
**Ingestion Status:** `SUCCESSFULLY LOGGED (H2A)`

Your safety profile has been fully cataloged inside the active conversation window:

| Safety Parameter | Value Configuration |
| :--- | :--- |
| **Full Name** | {data.get('name', 'N/A')} |
| **Job ID** | {data.get('job_id', 'N/A')} |
| **Employment Type** | {data.get('type', 'Employee')} |
| **Department** | {data.get('department', 'None Selected')} |
| **Specialised Skills** | {", ".join(data.get('specialised_skills', [])) or 'None'} |
| **Assigned Location(s)** | {", ".join(data.get('location', [])) or 'None'} |

---

### 🤝 Peer Agent Delegation Report (A2A Dispatch)
*The DoneSafe Coordinator Agent has dispatched task delegation payload `AUDIT-{data.get('job_id', '99999')}` to peer agent `store_compliance_auditor` for review.*

**Audit Agent Response Output:**
> "{auditor_agent_response}"
"""


def process_safety_record(form_state_json: str) -> str:
    """
    Saves the edited profile form state data to the safety records log, 
    and returns a clean markdown table playback response.
    """
    try:
        if isinstance(form_state_json, dict):
            data = form_state_json
        else:
            data = json.loads(form_state_json)
    except Exception:
        return "Error parsing form state. Ensure a valid payload configuration."

    if not data.get("name") or not data.get("job_id"):
        return "Validation Failed: 'Name' and 'Job ID' are mandatory safety fields."

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    log_entry = {
        "agent": "Safety Records Agent (DoneSafe)",
        "timestamp": timestamp,
        "record_payload": data
    }
    logger.info("A2UI DONE-SAFE RECORD INGESTION: %s", json.dumps(log_entry))

    return f"""
### [✓] DoneSafe Safety Record Logged Successfully
**Timestamp:** `{timestamp}`  
**Managed By:** `Safety Records Agent (DoneSafe)`

| Safety Record Parameter | Ingested Value Configuration |
| :--- | :--- |
| **Name** | {data.get('name')} |
| **Job ID** | {data.get('job_id')} |
| **Type** | {data.get('type', 'Employee')} |
| **Department** | {data.get('department', 'None Selected')} |
| **Specialised Skills** | {", ".join(data.get('specialised_skills', [])) or 'None Registered'} |
| **Assigned Location(s)** | {", ".join(data.get('location', [])) or 'None Registered'} |

#### Profile Snapshot (Rich Text Content Captured)
{data.get('profile_snapshot', '*(No content registered)*')}
"""

# -----------------------------------------------------------------------------
# Agent Registration Pipeline Block
# -----------------------------------------------------------------------------

available_tools = [render_safety_form, verify_latest_submission, process_safety_record]

root_agent = Agent(
    name="safety_records_agent_donesafe",
    model=build_gemini_model(
        model_path=FLASH_PATH, 
        temperature=0.1, 
        tools=available_tools
    ),
    include_contents='none',
    description="Handles rendering editable A2UI profile form catalogs, logging, and data record playbacks for DoneSafe entries.",
    tools=available_tools,
    instruction="""
    You are the specialized 'Safety Records Agent (DoneSafe)'. Your focus is managing the safety profile form loop.
    
    CRITICAL RENDERING RULE:
    When a user asks to see, open, build, or fill out the safety profile form, run the 'render_safety_form' tool immediately.
    Once you receive the live form URL, you MUST return a response guiding the user to split-screen the form on the right-hand panel of their workspace.
    
    Format your final conversational response EXACTLY like this (do not escape the brackets):
    "I have loaded your interactive Profile Management form!
    
    ### 🖥️ Option 1: Split-Screen Workspace (Form on the Right, Chat on the Left)
    You can view and interact with the form directly inside your playground workspace! Just click on the **Canvas** tab or the live form iframe rendered on the right-hand side of your Agentspace screen. This lets you fill out the form while we continue chatting right here!
    
    ### 🌐 Option 2: Clean Browser Tab
    If you prefer a full-screen experience, you can open the form in a separate browser tab by clicking this button:
    
    [![Open Interactive Form](https://img.shields.io/badge/OPEN_SAFETY_FORM-10B981?style=for-the-badge&logo=google&logoColor=white&labelColor=064E3B)](https://Ruchi-K.github.io/SHW-agent/)
    
    *Once you click submit in either window, simply close that view (or switch back here) and type **verify my submission** so I can display your captured data and dispatch the peer compliance audit!*"

    When the user says they have submitted the form, completed the entry, or asks you to check/verify their submission, run the 'verify_latest_submission' tool immediately to fetch and print the recorded details directly in the chat window.
    """
)

if __name__ == "__main__":
    print("[!] Standalone Safety Records Agent (DoneSafe) native execution harness active.")