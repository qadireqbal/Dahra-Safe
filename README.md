# Dhara-Safe

**Hyper-local flash-flood and landslide early-warning system for India.**

Dhara-Safe is a prototype that estimates flood and landslide risk at the **village/ward level**, explains why the risk is high, and provides emergency support such as SOS and nearby facilities.

The system combines weather, river, soil, terrain, earthquake and historical-event data with an ML pipeline to generate a live risk score for monitored locations.

> **Note:** This is a working prototype. Some external/live sources and historical datasets are simulated or use demo fallbacks where public real-time data is unavailable. These are clearly marked in the application.

## Features

- Village/ward-level flood and landslide risk scoring
- Explainable risk scores and contributing factors
- Rainfall forecasting with ARIMA
- Flood and landslide classification using Random Forest
- SCS-CN based runoff estimation
- Live weather, river, soil and earthquake data where available
- 15-minute risk update pipeline
- Historical model validation
- SOS and nearby emergency facilities
- Nearby-user emergency alerts
- Rule-based Dhara AI assistant
- Offline fallback using cached dashboard data
- Hindi/English starter support
- Razorpay payment integration for the prototype

## Folder Structure

```text
dhara-safe/
├── app.py
├── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── datasets/
│   ├── historical_training_data.csv
│   ├── validation_2025.csv
│   ├── dhara_safe_master_dataset.csv
│   └── locations_india.csv
└── pipeline/
    ├── locations.py
    ├── hydrology.py
    ├── feature_engineering.py
    ├── generate_dataset.py
    ├── train_model.py
    ├── backtest_historical.py
    └── models/
```

## Run Locally

Python **3.12** is recommended.

```bash
git clone <your-repo-url>
cd dhara-safe

python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset and train the models:

```bash
python -m pipeline.generate_dataset
python -m pipeline.train_model
```

Start the server:

```bash
uvicorn app:app --reload --port 8000
```

Then open:

```text
http://localhost:8000
```

The backend runs the risk pipeline automatically and updates monitored locations every 15 minutes.

## Data & Model

The included training and validation datasets are **synthetic/demo datasets** designed to test the complete ML pipeline.

The application can also use external sources such as:

- Open-Meteo
- Open-Meteo Flood API
- NASA POWER
- USGS earthquake data
- OpenStreetMap / Overpass

Availability depends on internet access and the relevant service.

For real-world deployment, the models should be retrained using verified historical records from appropriate Indian government and scientific sources.

## Current Limitations

This is a prototype, not a certified disaster-warning system.

- Some environmental data has demo fallbacks.
- Training data is synthetic.
- SMS requires an external provider/API key.
- SQLite should be replaced with PostgreSQL/PostGIS for production.
- Offline support is currently limited to cached dashboard data.
- Payment integration requires proper Razorpay configuration and compliance before handling real funds.

## Disclaimer

Dhara-Safe is a research/prototype project and should **not be used as the sole basis for emergency or evacuation decisions**.

For an actual deployment, the system would need verified datasets, operational data feeds, extensive field validation, monitoring, security review, and approval from the relevant authorities.
