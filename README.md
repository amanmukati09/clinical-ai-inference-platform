markdown# Clinical AI Inference Platform

Production-grade end-to-end system for diabetic patient 30-day readmission risk prediction.

## Architecture
Patient Data → Spring Boot API (8080)
↓
FastAPI ML Service (8000)
[PyTorch MLP Model]
↓
PostgreSQL (5432)
[Audit & Prediction Log]
↓
Next.js Frontend (3000)
[Risk Visualization UI]

**Design principle:** Separate ML inference from business logic. FastAPI owns model serving. Spring Boot owns audit logging, orchestration, and API contracts. PostgreSQL owns the source of truth.

## Quick Start

```bash
cd infrastructure/docker
docker compose up
```

Then open `http://localhost:3000`

## Dataset

UCI Diabetes 130-US Hospitals (1999–2008)
- 101,766 encounters, 42 features
- Target: 30-day readmission (11.2% positive class)
- Train/test split: **by patient** (79K / 20K), not by row
- Handles: 97% missing weight data, 95% missing A1C results

## Model

**PyTorch MLP** — 42 input features → [256, 128, 64] hidden layers → binary output

- **AUC:** 0.67 (consistent with published benchmarks on this dataset)
- **Recall:** 0.65 (prioritizes catching high-risk patients over precision)
- **Loss:** BCEWithLogitsLoss with pos_weight=8 (false negatives 8x more costly)

Why not XGBoost? This is a serving system, not a benchmark. PyTorch gives us:
- Consistent inference latency (19ms avg)
- Easy model versioning in Docker
- Production deployment patterns

## System Components

### ML Service (FastAPI, 8000)
- `/predict` — accepts patient encounter, returns risk score + confidence
- `/health` — liveness check for Kubernetes readiness probes
- Model artifacts loaded at startup (fail fast if corrupt)

### Backend (Spring Boot, 8080)
- `/api/v1/predict` — validates input, forwards to FastAPI, logs to PostgreSQL
- Every prediction gets a UUID requestId for tracing
- WebFlux WebClient for non-blocking calls

### Database (PostgreSQL)
- `prediction_records` table: full audit trail with input context, output, timestamp
- Enables drift detection, clinical review, compliance audits

### Frontend (Next.js, 3000)
- Single prediction form + risk visualization
- Shows risk category (LOW/MEDIUM/HIGH) with clinical thresholds
- Displays inference time, model version, confidence

## Design Decisions

### Why patient-level split?
Same patient can appear multiple times in the dataset (repeat encounters). Row-level split leaks patient information between train/test. Clinical reality: you predict on new patients, not revisits. Patient-level respects that.

### Why separate FastAPI from Spring Boot?
**Operational isolation:** ML crashes don't crash the API. Scale independently. Retrain without redeploying backend. Replace model with one-liner environment variable change.

### Why PostgreSQL logging for every prediction?
Regulated healthcare requires **explainability and auditability.** Every prediction must be traceable. Who requested it? When? What was the input? What did the model return? If a prediction fails, was it the model or the system?

### Why AUC 0.67?
This dataset is 25+ years old and missing critical modern features (social determinants, discharge instructions, post-discharge follow-up). Published research on the same data caps around AUC 0.72 with heavy feature engineering. **Our focus is systems depth, not benchmark chasing.** A production system with mediocre AUC beats a research model that doesn't deploy.

## Limitations & Future Work

**Known limitations:**
- Dataset from 1999–2008 (modern discharge processes differ)
- Missing social determinants (housing, transportation, support system)
- No temporal data (continuous monitoring, deterioration signals)

**Next steps:**
- Add XGBoost baseline for comparison
- Model performance monitoring (prediction volume, intervention rate trends)
- SHAP explainability for individual predictions
- A/B test against rule-based baseline

## Deployment

```bash
# Local development
cd ml-service && python src/training/trainer.py
cd backend && ./mvnw spring-boot:run
cd frontend && npm run dev

# Production (Docker)
docker compose up
```

Production considerations (not implemented):
- Load balancer in front of all 3 services
- Read replicas for PostgreSQL
- Model versioning / canary deployments
- Prometheus metrics scraping
- Structured logging (JSON, not console)

## Code Organization
ml-service/
src/
data/preprocessor.py        # Data pipeline: loading, cleaning, feature engineering
models/readmission_model.py  # PyTorch architecture
training/trainer.py          # Training loop, early stopping, evaluation
api/main.py                  # FastAPI server, /predict endpoint
backend/
src/main/java/com/clinicalai/platform/
controller/PredictionController.java
service/PredictionService.java        # Orchestration
service/MlServiceClient.java          # HTTP client to FastAPI
model/PredictionRecord.java           # JPA entity
repository/PredictionRepository.java  # PostgreSQL access
frontend/
app/page.tsx                   # Single-page prediction UI
infrastructure/docker/
docker-compose.yml             # 4-container orchestration
(Dockerfile files)

## Running Tests

```bash
cd ml-service
PYTHONPATH=src pytest tests/

cd backend
./mvnw test
```

## Clinical Notes

This system is a **decision-support tool only.** Final clinical decisions must be made by qualified healthcare professionals. Model trained on historical data; performance may differ on modern cohorts. Designed for care coordination triage, not autonomous decision-making.

---

**Author:** Aman | May 2026
