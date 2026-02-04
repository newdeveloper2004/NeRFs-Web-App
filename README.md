# NeRF Web App 🎥

A web application that generates 360° GIF animations from Neural Radiance Fields (NeRF) models.

![NeRF](https://img.shields.io/badge/NeRF-Neural%20Radiance%20Fields-cyan)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![React](https://img.shields.io/badge/React-Frontend-blue)

## Features

- 🎯 Generate 360° turntable animations from trained NeRF models
- 🖼️ Real-time GIF preview in browser
- ⬇️ Download rendered GIF animations
- 🎨 Modern, responsive UI built with React and Tailwind CSS

## Project Structure

```
NERF-V1/
├── README.md
├── requirements.txt
│
├── Model/                   # NeRF Model Files
│   ├── NeRF.py             # NeRF architecture
│   └── nerf_model.pth      # Trained model weights
│
└── Web App/
    ├── backend/            # FastAPI Backend
    │   ├── main.py         # API endpoints
    │   ├── database.py     # SQLite database config
    │   ├── models.py       # SQLAlchemy models
    │   ├── schemas.py      # Pydantic schemas
    │   ├── generate_gif.py # NeRF rendering logic
    │   └── static/         # Built React frontend
    │
    └── frontend/           # React Frontend
        ├── src/
        │   └── pages/Homepage.jsx
        ├── package.json
        └── vite.config.js
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your NeRF Model

Copy your trained `nerf_model.pth` file to the `Web App/backend/` folder.

### 3. Run the Server

```bash
cd "Web App/backend"
python -m uvicorn main:app --reload
```

### 4. Open in Browser

Navigate to **http://localhost:8000**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend |
| `POST` | `/render` | Start new render job |
| `GET` | `/render/{job_id}/status` | Get job status |
| `GET` | `/render/{job_id}/download` | Download GIF |
| `GET` | `/renders` | List all render jobs |
| `DELETE` | `/render/{job_id}` | Delete render job |

## Rebuilding the Frontend

If you modify the React frontend:

```bash
cd "Web App/frontend"
npm install
npm run build
Copy-Item -Recurse -Force ".\dist\*" "..\backend\static\"
```

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: React, Vite, Tailwind CSS
- **ML**: PyTorch, Neural Radiance Fields

## License

MIT License

----


