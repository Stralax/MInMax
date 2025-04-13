from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import subprocess

# Initialize the app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/coffee", response_class=PlainTextResponse)
def download_image(image_url: str = Query(...)):
    result = subprocess.run(
        ["python3", "scripts/coffee.py", image_url],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return result.stdout
    else:
        return result.stderr

@app.get("/answer", response_class=PlainTextResponse)
def download_image(text: str = Query(...)):
    result = subprocess.run(
        ["python3", "scripts/answers.py", text],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return result.stdout
    else:
        return result.stderr

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)