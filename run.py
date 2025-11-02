import uvicorn
import main # Import your main application file

if __name__ == "__main__":
    # Launch Uvicorn using the standard Python script execution method.
    # This prevents the multiprocessing conflicts you are seeing.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)