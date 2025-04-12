from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

# Inicializiraj aplikacijo
app = FastAPI()

image_url="https://media.cnn.com/api/v1/images/stellar/prod/150929101049-black-coffee-stock.jpg?q=w_3000,h_3074,x_0,y_0,c_fill"

from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse  # Ensure this is imported
import subprocess

app = FastAPI()

# url za api klic: http://127.0.0.1:8000/test?image_url=https://media.cnn.com/api/v1/images/stellar/prod/150929101049-black-coffee-stock.jpg?q=w_3000,h_3074,x_0,y_0,c_fill
# api dobi url slike in vrne besedilo
@app.get("/coffee", response_class=PlainTextResponse)
def download_image(image_url: str = Query(...)):  # Expecting image_url as a query parameter
    # Run the external Python script with the image_url as an argument
    result = subprocess.run(
        ["python3", "scripts/coffee.py", image_url],  # Calling your script with the image_url as argument
        capture_output=True,
        text=True
    )
    # Check if the script ran successfully and return the output
    if result.returncode == 0:
        return result.stdout  # Return the output from the script
    else:
        return result.stderr  # Return any errors if the script fails

@app.get("/answer", response_class=PlainTextResponse)
def download_image(text: str = Query(...)):
    result = subprocess.run(
        ["python3", "scripts/answers.py", text],
        capture_output=True,
        text=True
    )
    # Check if the script ran successfully and return the output
    if result.returncode == 0:
        return result.stdout
    else:
        return result.stderr

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)