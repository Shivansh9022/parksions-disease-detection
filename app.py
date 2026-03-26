"""
NeuroScan — Parkinson's Detection Backend
==========================================
Flask API that loads best_model.pkl (SVC, rbf, C=100, γ=0.1)
and exposes:

  POST /predict — JSON array of 22 features → prediction + confidence
  GET  /status  — model readiness check

The scaler is fitted once at startup using the same random_state=2
split logic from training. You must place Parkinsson_disease.csv
alongside app.py OR set PARKINSONS_CSV env var pointing to it.

Run:
    pip install flask flask-cors scikit-learn numpy pandas
    python app.py
"""

import io, os, pickle, warnings
import numpy  as np
import pandas as pd
from   flask       import Flask, request, jsonify, send_from_directory
from   flask_cors  import CORS
from   sklearn.preprocessing  import StandardScaler
from   sklearn.metrics        import accuracy_score
from   sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ── Feature column order (must match training) ──────────────────────────────
FEATURE_COLS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
    "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
CSV_PATH   = os.environ.get(
    "PARKINSONS_CSV",
    os.path.join(BASE_DIR, "Parkinsson disease.csv")
)

# ── Load pre-trained model ──────────────────────────────────────────────────
print("[NeuroScan] Loading model …")
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

print(f"[NeuroScan] Model OK → {MODEL.__class__.__name__} "
      f"kernel={MODEL.kernel} C={MODEL.C} γ={MODEL.gamma} "
      f"n_features={MODEL.n_features_in_}")

# ── Fit scaler from dataset CSV at startup ──────────────────────────────────
SCALER     = None
TRAIN_ACC  = None
TEST_ACC   = None
DATASET    = {}

def _fit_scaler():
    global SCALER, TRAIN_ACC, TEST_ACC, DATASET

    if not os.path.exists(CSV_PATH):
        print(f"[NeuroScan] ⚠  Dataset CSV not found at: {CSV_PATH}")
        print("[NeuroScan]    Set PARKINSONS_CSV env var or place the CSV next to app.py")
        print("[NeuroScan]    Predictions will fail until scaler is ready.")
        return

    df = pd.read_csv(CSV_PATH)
    X  = df[FEATURE_COLS].values
    y  = df["status"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    SCALER     = StandardScaler()
    X_train_s  = SCALER.fit_transform(X_train)
    X_test_s   = SCALER.transform(X_test)

    TRAIN_ACC  = round(accuracy_score(y_train, MODEL.predict(X_train_s)) * 100, 2)
    TEST_ACC   = round(accuracy_score(y_test,  MODEL.predict(X_test_s))  * 100, 2)

    DATASET = {
        "total":    int(len(df)),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
    }

    print(f"[NeuroScan] Scaler ready | Train: {TRAIN_ACC}%  Test: {TEST_ACC}%")

_fit_scaler()


# ── Helpers ─────────────────────────────────────────────────────────────────
def sigmoid(x):
    return float(1 / (1 + np.exp(-x)))


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/status")
def status():
    return jsonify({
        "model_loaded": True,
        "scaler_ready": SCALER is not None,
        "model_type":   MODEL.__class__.__name__,
        "kernel":       MODEL.kernel,
        "C":            MODEL.C,
        "gamma":        MODEL.gamma,
        "n_features":   MODEL.n_features_in_,
        "train_acc":    TRAIN_ACC,
        "test_acc":     TEST_ACC,
        "dataset":      DATASET,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if SCALER is None:
        return jsonify({"error": "Model not ready. Place the CSV next to app.py and restart."}), 503

    data = request.get_json(force=True)
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400

    features = data["features"]
    if len(features) != MODEL.n_features_in_:
        return jsonify({
            "error": f"Expected {MODEL.n_features_in_} features, got {len(features)}"
        }), 400

    try:
        X        = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = SCALER.transform(X)

        prediction = int(MODEL.predict(X_scaled)[0])
        dec_score  = float(MODEL.decision_function(X_scaled)[0])
        confidence = round(sigmoid(abs(dec_score)) * 100, 2)
        label      = "Parkinson's Detected" if prediction == 1 else "No Parkinson's Detected"

        return jsonify({
            "prediction":     prediction,
            "label":          label,
            "confidence":     confidence,
            "decision_score": round(dec_score, 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════╗")
    print("║   NeuroScan  —  http://localhost:5000    ║")
    print("╚══════════════════════════════════════════╝\n")
    app.run(debug=True, port=5000)