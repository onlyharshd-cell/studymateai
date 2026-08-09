from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from datetime import date, datetime, timedelta
from typing import List, Optional
import os

import json
import sqlite3
import hashlib
import secrets
import time


# ============================================================
# GEMINI API CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured on the backend.")

GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# AUTHENTICATION / SQLITE
# ============================================================

DATABASE = "/tmp/studymate.db"
SESSION_DAYS = 7


def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    connection = get_db()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT NOT NULL UNIQUE,
            expires_at INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    connection.commit()
    connection.close()


init_database()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        210_000
    )

    return (
        salt.hex()
        + ":"
        + password_hash.hex()
    )


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, hash_hex = stored_hash.split(":", 1)

        salt = bytes.fromhex(salt_hex)

        expected_hash = bytes.fromhex(hash_hex)

        actual_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            210_000
        )

        return secrets.compare_digest(
            actual_hash,
            expected_hash
        )

    except (ValueError, TypeError):
        return False


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(48)

    now = int(time.time())

    expires_at = now + (
        SESSION_DAYS * 24 * 60 * 60
    )

    connection = get_db()

    connection.execute(
        """
        INSERT INTO sessions
        (user_id, token, expires_at, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            token,
            expires_at,
            now
        )
    )

    connection.commit()
    connection.close()

    return token


def get_user_from_token(token: Optional[str]):
    if not token:
        return None

    now = int(time.time())

    connection = get_db()

    row = connection.execute(
        """
        SELECT
            users.id,
            users.name,
            users.email
        FROM sessions
        JOIN users
            ON users.id = sessions.user_id
        WHERE sessions.token = ?
          AND sessions.expires_at > ?
        """,
        (token, now)
    ).fetchone()

    # Remove expired sessions while we're here.
    connection.execute(
        """
        DELETE FROM sessions
        WHERE expires_at <= ?
        """,
        (now,)
    )

    connection.commit()
    connection.close()

    if row is None:
        return None

    return dict(row)


def get_bearer_token(
    authorization: Optional[str]
) -> Optional[str]:

    if not authorization:
        return None

    parts = authorization.split(
        " ",
        1
    )

    if len(parts) != 2:
        return None

    scheme, token = parts

    if scheme.lower() != "bearer":
        return None

    return token.strip() or None


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="StudyMate AI",
    description="AI Study Planner for College Students",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://steady-mousse-0ae4ab.netlify.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AUTHENTICATION MODELS
# ============================================================

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=200)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=200)
    password: str = Field(..., min_length=6, max_length=128)


class AuthUser(BaseModel):
    id: int
    name: str
    email: str


class AuthResponse(BaseModel):
    success: bool
    message: str
    access_token: str
    user: AuthUser


# ============================================================
# REQUEST MODELS
# ============================================================

class SubjectInput(BaseModel):

    name: str = Field(
        ...,
        min_length=1,
        max_length=100
    )

    difficulty: str = "medium"

    confidence: int = Field(
        default=3,
        ge=1,
        le=5
    )

    topics: List[str] = Field(
        default_factory=list
    )


class StudyPlanRequest(BaseModel):

    goal: str = Field(
        ...,
        min_length=1,
        max_length=300
    )

    exam_date: date

    daily_hours: float = Field(
        ...,
        ge=1,
        le=16
    )

    subjects: List[SubjectInput] = Field(
        ...,
        min_length=1,
        max_length=20
    )

    study_days: List[str] = Field(
        default_factory=lambda: [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    preferred_session: str = "flexible"

    include_revision: bool = True

    include_practice: bool = True


# ============================================================
# RESPONSE MODELS
# ============================================================

class SubjectPriority(BaseModel):

    subject: str

    priority: str

    reason: str

    recommended_hours: float


class StudyTask(BaseModel):

    title: str

    type: str

    duration_minutes: int

    subject: str

    completed: bool = False


class DailyPlan(BaseModel):

    day: int

    date: str

    focus: str

    total_minutes: int

    tasks: List[StudyTask]


class StudyPlanResponse(BaseModel):

    goal: str

    exam_date: str

    days_remaining: int

    daily_hours: float

    total_planned_hours: float

    subject_priorities: List[SubjectPriority]

    daily_plan: List[DailyPlan]

    strategy: List[str]

    ai_recommendation: str


# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

@app.post("/api/auth/register", response_model=AuthResponse)
def register(request: RegisterRequest):

    name = request.name.strip()
    email = normalize_email(request.email)

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name is required."
        )

    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid email address."
        )

    connection = get_db()

    existing_user = connection.execute(
        """
        SELECT id
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    if existing_user:
        connection.close()

        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    password_hash = hash_password(
        request.password
    )

    created_at = date.today().isoformat()

    cursor = connection.execute(
        """
        INSERT INTO users
        (name, email, password_hash, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            name,
            email,
            password_hash,
            created_at
        )
    )

    user_id = cursor.lastrowid

    connection.commit()
    connection.close()

    token = create_session(user_id)

    return {
        "success": True,
        "message": "Account created successfully.",
        "access_token": token,
        "user": {
            "id": user_id,
            "name": name,
            "email": email
        }
    }


@app.post("/api/auth/login", response_model=AuthResponse)
def login(request: LoginRequest):

    email = normalize_email(
        request.email
    )

    connection = get_db()

    user = connection.execute(
        """
        SELECT
            id,
            name,
            email,
            password_hash
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    connection.close()

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    if not verify_password(
        request.password,
        user["password_hash"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    token = create_session(
        user["id"]
    )

    return {
        "success": True,
        "message": "Login successful.",
        "access_token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"]
        }
    }


@app.get("/api/auth/me")
def current_user(
    authorization: Optional[str] = Header(default=None)
):

    token = get_bearer_token(
        authorization
    )

    user = get_user_from_token(
        token
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated."
        )

    return {
        "success": True,
        "user": user
    }


@app.post("/api/auth/logout")
def logout(
    authorization: Optional[str] = Header(default=None)
):

    token = get_bearer_token(
        authorization
    )

    if not token:
        return {
            "success": True,
            "message": "Already logged out."
        }

    connection = get_db()

    connection.execute(
        """
        DELETE FROM sessions
        WHERE token = ?
        """,
        (token,)
    )

    connection.commit()
    connection.close()

    return {
        "success": True,
        "message": "Logged out successfully."
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "StudyMate AI Backend is running 🚀",
        "framework": "FastAPI",
        "ai": "Gemini",
        "model": GEMINI_MODEL
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "service": "StudyMate AI",
        "ai_provider": "Gemini",
        "model": GEMINI_MODEL,
        "authentication": "enabled",
        "database": DATABASE
    }


# ============================================================
# GEMINI TEST
# ============================================================

@app.get("/api/test-gemini")
def test_gemini():

    try:

        response = client.models.generate_content(

            model=GEMINI_MODEL,

            contents=(
                "Reply with exactly: "
                "StudyMate Gemini connection working."
            )
        )

        return {
            "success": True,
            "message": response.text
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# SUBJECT INFORMATION
# ============================================================

def build_subject_information(subjects):

    result = ""

    for subject in subjects:

        topics = (
            ", ".join(subject.topics)
            if subject.topics
            else "Topics not provided"
        )

        result += f"""

Subject: {subject.name}

Difficulty: {subject.difficulty}

Confidence: {subject.confidence}/5

Topics: {topics}

"""

    return result


# ============================================================
# STUDY PLANNER
# ============================================================

@app.post(
    "/api/study-plan",
    response_model=StudyPlanResponse
)
def create_study_plan(
    request: StudyPlanRequest,
    authorization: Optional[str] = Header(default=None)
):

    # Optional authentication:
    # If the frontend sends Authorization: Bearer <token>,
    # we can identify the student. The planner remains backward
    # compatible with the current HTML, which does not yet send
    # a token.
    token = get_bearer_token(authorization)

    logged_in_user = get_user_from_token(token)

    today = date.today()


    # --------------------------------------------------------
    # Calculate remaining days
    # --------------------------------------------------------

    days_remaining = (
        request.exam_date - today
    ).days


    if days_remaining <= 0:

        raise HTTPException(
            status_code=400,
            detail="Exam date must be in the future."
        )


    # Don't generate an enormous response.

    plan_days = min(
        days_remaining,
        60
    )


    daily_minutes_limit = int(
        request.daily_hours * 60
    )


    subjects = build_subject_information(
        request.subjects
    )


    # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = f"""

You are StudyMate AI.

You are an expert academic study planner
for college students.

Create a personalized, realistic study plan.

TODAY:
{today.isoformat()}

STUDENT GOAL:
{request.goal}

EXAM DATE:
{request.exam_date.isoformat()}

DAYS REMAINING:
{days_remaining}

DAILY STUDY TIME:
{request.daily_hours} hours

MAXIMUM DAILY STUDY MINUTES:
{daily_minutes_limit}

PREFERRED STUDY STYLE:
{request.preferred_session}

INCLUDE REVISION:
{request.include_revision}

INCLUDE PRACTICE:
{request.include_practice}

SUBJECTS:

{subjects}


PLANNING RULES:

1. Prioritize difficult subjects.

2. Give more attention to subjects
   where confidence is low.

3. Do not exceed {daily_minutes_limit}
   study minutes in any day.

4. Each study session should normally
   be between 30 and 90 minutes.

5. Include learning sessions.

6. Include practice sessions.

7. Include revision sessions.

8. Use spaced revision.

9. Include mock tests closer to the exam.

10. Do not put all difficult subjects
    on the same day.

11. Keep the final days focused mainly
    on revision and practice.

12. Make the schedule realistic for
    a college student.

13. Breaks should not count as study time.

14. Never create a task with 0 minutes.

15. Never create negative durations.

Generate a maximum of {plan_days} days.


RETURN ONLY VALID JSON.

Use exactly this structure:

{{
    "goal": "{request.goal}",

    "exam_date": "{request.exam_date.isoformat()}",

    "daily_hours": {request.daily_hours},

    "subject_priorities": [

        {{
            "subject": "Data Structures",
            "priority": "High",
            "reason": "Low confidence and high difficulty",
            "recommended_hours": 30
        }}

    ],

    "daily_plan": [

        {{
            "day": 1,

            "date": "{today.isoformat()}",

            "focus": "Data Structures",

            "total_minutes": 240,

            "tasks": [

                {{
                    "title": "Study Binary Trees",
                    "type": "learning",
                    "duration_minutes": 60,
                    "subject": "Data Structures",
                    "completed": false
                }}

            ]
        }}

    ],

    "strategy": [

        "Use active recall.",
        "Practice questions after learning.",
        "Revise difficult topics regularly."

    ],

    "ai_recommendation":
        "Focus on your weakest subject first."
}}


ALLOWED TASK TYPES:

learning
practice
revision
mock_test
weak_area
break

IMPORTANT:

Return JSON only.

Do not use markdown.

Do not write anything outside the JSON.
"""


    # ========================================================
    # CALL GEMINI
    # ========================================================

    try:

        response = client.models.generate_content(

            model=GEMINI_MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(

                response_mime_type="application/json"

            )
        )


        if not response.text:

            raise Exception(
                "Gemini returned an empty response."
            )


        raw = response.text.strip()


        # Remove accidental markdown fences.

        if raw.startswith("```"):

            raw = (
                raw
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )


        data = json.loads(raw)


        # ====================================================
        # SERVER VALIDATION
        # ====================================================

        total_minutes = 0


        for day in data.get(
            "daily_plan",
            []
        ):

            day_minutes = 0


            for task in day.get(
                "tasks",
                []
            ):

                duration = int(
                    task.get(
                        "duration_minutes",
                        0
                    )
                )


                if duration <= 0:

                    raise Exception(
                        "Gemini generated an invalid task duration."
                    )


                if task.get(
                    "type"
                ) != "break":

                    day_minutes += duration


            # Prevent impossible schedules.

            if day_minutes > daily_minutes_limit:

                raise Exception(
                    f"Day {day.get('day')} exceeds "
                    "the student's daily study limit."
                )


            day["total_minutes"] = (
                day_minutes
            )


            total_minutes += (
                day_minutes
            )


        # ====================================================
        # CALCULATE TOTAL HOURS OURSELVES
        # ====================================================

        total_hours = round(
            total_minutes / 60,
            2
        )


        data["goal"] = request.goal

        data["exam_date"] = (
            request.exam_date.isoformat()
        )

        data["days_remaining"] = (
            days_remaining
        )

        data["daily_hours"] = (
            request.daily_hours
        )

        data["total_planned_hours"] = (
            total_hours
        )


        return data


    except json.JSONDecodeError:

        raise HTTPException(

            status_code=500,

            detail=(
                "Gemini returned invalid JSON. "
                "Please try generating the plan again."
            )
        )


    except HTTPException:

        raise


    except Exception as error:

        print(
            "\n=============================="
        )

        print(
            "GEMINI / PLANNER ERROR"
        )

        print(
            "=============================="
        )

        print(
            repr(error)
        )

        print(
            "==============================\n"
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to generate study plan. "
                "Check your Gemini API key and "
                "model configuration."
            )
        )

# ================= ADAPTIVE STUDY FEATURES =================

def ensure_extra_tables():
    db = get_db()
    db.execute('CREATE TABLE IF NOT EXISTS study_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER,subject TEXT,topic TEXT,minutes INTEGER,confidence_before INTEGER,confidence_after INTEGER,created_at TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS quiz_results (id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER,subject TEXT,topic TEXT,score INTEGER,total INTEGER,created_at TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS revision_items (id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER,subject TEXT,topic TEXT,next_revision TEXT,interval_days INTEGER,mastery REAL)')
    db.commit(); db.close()

ensure_extra_tables()

def ai_json(prompt: str):
    if client is None:
        raise HTTPException(503, 'Gemini is not configured. Add GEMINI_API_KEY to the backend.')
    try:
        r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=types.GenerateContentConfig(response_mime_type='application/json'))
        raw = (r.text or '').strip().replace('```json','').replace('```','').strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(502, 'Gemini returned invalid JSON.')
    except Exception as e:
        print('Gemini:', repr(e)); raise HTTPException(502, 'Gemini request failed.')

class QuizRequest(BaseModel):
    subject: str
    topic: str
    count: int = Field(5, ge=3, le=10)
    difficulty: str = 'medium'

class QuizSubmit(BaseModel):
    subject: str
    topic: str
    score: int = Field(..., ge=0)
    total: int = Field(..., ge=1)

class StudySessionRequest(BaseModel):
    subject: str
    topic: str
    minutes: int = Field(..., ge=1, le=300)
    confidence_before: int = Field(3, ge=1, le=5)
    confidence_after: int = Field(4, ge=1, le=5)

class TeachRequest(BaseModel):
    subject: str
    topic: str
    level: str = 'exam'

class NextActionRequest(BaseModel):
    goal: str = ''
    exam_date: Optional[date] = None

class RevisionRequest(BaseModel):
    subject: str
    topic: str
    mastery: float = Field(.5, ge=0, le=1)

@app.get('/api/knowledge-map')
def knowledge_map(authorization: Optional[str] = Header(default=None)):
    user = require_user(authorization)
    db = get_db()
    rows = db.execute('SELECT name,difficulty,confidence,mastery,total_quiz_questions,correct_quiz_questions,total_study_minutes FROM subjects WHERE user_id=?', (user['id'],)).fetchall()
    db.close()
    out=[]
    for x in rows:
        accuracy = x['correct_quiz_questions']/x['total_quiz_questions'] if x['total_quiz_questions'] else 0
        mastery = max(float(x['mastery'] or 0), accuracy)
        if not mastery: mastery=max(.1, x['confidence']/5*.55)
        out.append({'subject':x['name'],'difficulty':x['difficulty'],'confidence':x['confidence'],'mastery':round(min(mastery,1),2),'quiz_accuracy':round(accuracy,2),'study_minutes':x['total_study_minutes']})
    return {'subjects':out}

@app.post('/api/next-action')
def next_action(req: NextActionRequest, authorization: Optional[str] = Header(default=None)):
    user=require_user(authorization)
    db=get_db(); rows=db.execute('SELECT name,difficulty,confidence,mastery,total_quiz_questions,correct_quiz_questions,total_study_minutes FROM subjects WHERE user_id=?',(user['id'],)).fetchall(); db.close()
    if not rows:
        return {'title':'Add your subjects','reason':'Add subjects and confidence levels so StudyMate can choose your next action.','duration_minutes':20,'subject':'Setup','topic':'Subjects'}
    days=(req.exam_date-date.today()).days if req.exam_date else 30
    prompt=f'''Choose the single best next study action for a college student. Goal: {req.goal}. Days: {days}. Performance: {json.dumps([dict(x) for x in rows])}. Return JSON only: {{"title":"...","reason":"...","duration_minutes":45,"subject":"...","topic":"...","action_type":"learning|practice|revision|quiz|weak_area"}}'''
    return ai_json(prompt)

@app.post('/api/quiz/generate')
def quiz_generate(req: QuizRequest, authorization: Optional[str] = Header(default=None)):
    require_user(authorization)
    prompt=f'''Create {req.count} college-level MCQs for {req.subject} / {req.topic}. Difficulty: {req.difficulty}. Return JSON only: {{"subject":"{req.subject}","topic":"{req.topic}","questions":[{{"id":1,"question":"...","options":["A","B","C","D"],"answer":0,"explanation":"..."}}]}}'''
    return ai_json(prompt)

@app.post('/api/quiz/submit')
def quiz_submit(req: QuizSubmit, authorization: Optional[str] = Header(default=None)):
    user=require_user(authorization); db=get_db()
    db.execute('INSERT INTO quiz_results(user_id,subject,topic,score,total,created_at) VALUES(?,?,?,?,?,?)',(user['id'],req.subject,req.topic,req.score,req.total,datetime.utcnow().isoformat()))
    row=db.execute('SELECT * FROM subjects WHERE user_id=? AND name=?',(user['id'],req.subject)).fetchone()
    acc=req.score/req.total
    if row:
        nt=row['total_quiz_questions']+req.total; nc=row['correct_quiz_questions']+req.score; mastery=round(.75*(nc/nt)+.25*(row['confidence']/5),3)
        db.execute('UPDATE subjects SET total_quiz_questions=?,correct_quiz_questions=?,mastery=? WHERE user_id=? AND name=?',(nt,nc,mastery,user['id'],req.subject))
    else:
        mastery=acc; db.execute('INSERT INTO subjects(user_id,name,confidence,mastery,total_quiz_questions,correct_quiz_questions) VALUES(?,?,?,?,?,?)',(user['id'],req.subject,3,mastery,req.total,req.score))
    db.commit(); db.close()
    return {'score':req.score,'total':req.total,'accuracy':round(acc,2),'mastery':round(mastery,2),'message':'Weak area detected. Revisit this topic soon.' if acc<.6 else 'Good progress. Keep it in spaced revision.'}

@app.post('/api/session')
def study_session(req: StudySessionRequest, authorization: Optional[str] = Header(default=None)):
    user=require_user(authorization); db=get_db()
    db.execute('INSERT INTO study_sessions(user_id,subject,topic,minutes,confidence_before,confidence_after,created_at) VALUES(?,?,?,?,?,?,?)',(user['id'],req.subject,req.topic,req.minutes,req.confidence_before,req.confidence_after,datetime.utcnow().isoformat()))
    db.execute('UPDATE subjects SET total_study_minutes=total_study_minutes+? WHERE user_id=? AND name=?',(req.minutes,user['id'],req.subject)); db.commit(); db.close()
    return {'success':True}

@app.post('/api/teach')
def teach(req: TeachRequest, authorization: Optional[str] = Header(default=None)):
    require_user(authorization)
    return ai_json(f'''Teach a college student {req.topic} in {req.subject}, level {req.level}. Return JSON only: {{"title":"...","explanation":"...","example":"...","common_mistake":"...","checkpoint":"...","next_step":"..."}}''')

@app.post('/api/revision/schedule')
def revision(req: RevisionRequest, authorization: Optional[str] = Header(default=None)):
    user=require_user(authorization)
    interval=1 if req.mastery<.4 else 2 if req.mastery<.6 else 4 if req.mastery<.8 else 7
    nxt=date.today()+timedelta(days=interval)
    db=get_db(); db.execute('INSERT INTO revision_items(user_id,subject,topic,next_revision,interval_days,mastery) VALUES(?,?,?,?,?,?)',(user['id'],req.subject,req.topic,nxt.isoformat(),interval,req.mastery)); db.commit(); db.close()
    return {'next_revision':nxt.isoformat(),'interval_days':interval}

@app.get('/api/revision/due')
def revision_due(authorization: Optional[str] = Header(default=None)):
    user=require_user(authorization); db=get_db(); rows=db.execute('SELECT subject,topic,next_revision,interval_days,mastery FROM revision_items WHERE user_id=? AND next_revision<=? ORDER BY next_revision',(user['id'],date.today().isoformat())).fetchall(); db.close()
    return {'items':[dict(x) for x in rows]}

@app.post('/api/syllabus/analyze')
async def syllabus_analyze(file: UploadFile = File(...), authorization: Optional[str] = Header(default=None)):
    require_user(authorization)
    data=await file.read(); text=''
    if (file.filename or '').lower().endswith('.pdf'):
        try:
            from pypdf import PdfReader
            import io
            text='\n'.join((p.extract_text() or '') for p in PdfReader(io.BytesIO(data)).pages)
        except Exception:
            raise HTTPException(400,'Could not read this PDF.')
    else:
        text=data.decode('utf-8',errors='ignore')
    if not text.strip(): raise HTTPException(400,'No readable text found.')
    prompt=f'''Analyze this college syllabus or PYQ text. Return JSON only: {{"title":"...","high_priority_topics":["..."],"suggested_first_topics":["..."],"units":[{{"unit":"Unit 1","topics":["..."],"priority":"High","reason":"..."}}],"exam_strategy":"..."}}\nSOURCE:\n{text[:100000]}'''
    return ai_json(prompt)

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False
    )
