import uvicorn
from fastapi import FastAPI

# Create FastAPI application instance
app = FastAPI()


if __name__ == '__main__':
    # Run the FastAPI app using Uvicorn with auto-reload (for development)
    uvicorn.run("fast_api:app", host="127.0.0.1", port=8000, reload=True)



