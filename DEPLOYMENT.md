# Deployment

## Model artifact and safety

The inference service loads `models/disease_model.pkl` once when
`app.model` is imported. The path is resolved relative to the source tree,
not the current working directory. This is a joblib/pickle serialization of
the existing binary Random Forest and must remain a trusted, reviewed
project artifact. Python pickle/joblib loading can execute code: never load
an uploaded, user-supplied, or otherwise untrusted model file.

The active artifact SHA-256 is
`1ACA8A37CE0CD4C070D0E9F4442AC698086D61AF699AB4C1B18F2E591A5C0888`.
`models/disease_model.joblib` is retained but is not used by the service.
The Docker image copies only `disease_model.pkl`. Training does not replace
the active artifact unless an explicit output path is provided; review and
validate candidate artifacts before changing the deployment input.

No runtime secrets or external data are required. The bundled 20 records are
synthetic only. Never submit actual patient data; this academic prototype is
not clinically validated and must not be used for diagnosis, triage, or
treatment.

## Local Windows PowerShell

From the repository root:

```powershell
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The service is available at `http://127.0.0.1:8000`; health is
`http://127.0.0.1:8000/health` and OpenAPI documentation is
`http://127.0.0.1:8000/docs`. Use `--host 0.0.0.0` only when the service
must accept connections from another container or host; do not expose it
publicly as a replacement for the Spring Boot API.

Run checks before deploying:

```powershell
.\venv\Scripts\python.exe -m pytest -v
.\venv\Scripts\python.exe -m app.train_model
```

Training prints reproducible split metrics and does not overwrite the
inference model. Tests use only the bundled synthetic records.

## Docker

Build and start from the repository root:

```powershell
docker build -t edps-ml-service:local .
docker run --rm --name edps-ml-service -p 8000:8000 edps-ml-service:local
```

The image uses Python 3.13, installs the pinned packages from
`requirements.txt`, runs Uvicorn on port 8000 as an unprivileged user, and
defines a health check that requires the model to be ready. Container health
is visible through `docker inspect`; the HTTP health route remains `/health`.

For production, publish the image to a private registry and place Spring Boot
and FastAPI on a private network. Terminate public TLS and authentication at
the application's ingress/API layer. Do not enable browser-to-FastAPI CORS
access as an integration shortcut. Configure Spring Boot connect/read
timeouts as described in `ML_API_CONTRACT.md`; allow at least 60 seconds for
the first global-importance request or warm it before accepting dashboard
traffic. Importance and FedAvg caches are process-local; each additional
worker/container computes its own first result.

## Dependency and CI compatibility

Exact tested package versions are pinned in `requirements.txt`. GitHub
Actions uses Python 3.13 and runs `python -m pytest -v` on pushes and pull
requests. Re-run the full suite after changing dependencies or replacing
the model artifact.
