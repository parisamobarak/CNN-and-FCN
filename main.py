import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI
app = FastAPI()

from CNN import router as CNN_router
app.include_router(CNN_router)

from FCN import router as FCN_router
app.include_router(FCN_router)