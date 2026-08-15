# backend/server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.ovf import router as ovf_router, warmup as ovf_warmup
from api.h5 import router as h5_router
from api.preproc import router as preproc_router

app = FastAPI(title="Spine AI Platform API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ovf_router)
app.include_router(h5_router)
app.include_router(preproc_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def _startup():
    await ovf_warmup()
    print("Routers mounted & OVF warm-up done")


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000)
