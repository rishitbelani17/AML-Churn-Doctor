# Churn Doctor - Setup & Run Guide

## ✅ IMPORTANT: No API Keys Needed!

**All OpenAI code has been replaced with free Hugging Face models that run locally.**
- ❌ **NO** `OPENAI_API_KEY` needed
- ❌ **NO** API costs
- ✅ Models download automatically on first use (one-time, ~1-2GB)
- ✅ Everything runs locally on your machine

---

## Quick Start Guide

### Step 1: Activate Virtual Environment

```bash
# Navigate to project directory (if not already there)
cd "/Users/aryarohitshidore/Desktop/AML/AML-Churn-Doctor/AML - CHURN DOCTOR"

# Activate virtual environment
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First installation may take 5-10 minutes as it downloads:
- PyTorch (~2GB)
- Hugging Face models (~1-2GB total)

### Step 3: Generate Sample Data

```bash
python generate_data.py
```

This creates the `churn_doctor_data/` directory with sample CSV files:
- `businesses.csv`
- `customers.csv`
- `orders.csv`
- `interactions.csv`

### Step 4: Train the Churn Model

```bash
python churnmodel.py
```

This will:
- Train an XGBoost model on the data
- Save the model to `models/churn_model.pkl`
- Print validation AUC score

**Expected output:**
```
Validation AUC: 0.XXX
Model saved
```

### Step 5: Run the Dashboard

```bash
source venv/bin/activate && streamlit run dashboard.py 2>&1 | head -50
```

The dashboard will automatically open in your browser at:
**http://localhost:8501**

You can:
- Select different businesses from the sidebar
- View churn scores, metrics, and insights
- See SHAP feature importances
- View complaint analysis

---

## Optional: Email Configuration

**Only needed if you want to use the email notification feature.**

Edit `emailer.py` and replace:

```python
SMTP_USER = "YOUR_EMAIL@gmail.com"  # Replace with your Gmail
SMTP_PASS = "YOUR_APP_PASSWORD"      # Replace with Gmail App Password
```

### How to Get Gmail App Password:

1. Go to your Google Account settings
2. Enable 2-Step Verification
3. Go to "App passwords"
4. Generate a new app password for "Mail"
5. Use that password in `emailer.py`

**Note:** If you don't configure email, the `schedular_job.py` will fail when trying to send emails, but the dashboard and other features will work fine.

---

## Running Scheduled Jobs (Optional)

```bash
python schedular_job.py
```

This will:
- Generate insights for a business
- Create an email report
- Send it via email (if configured)

You can edit `schedular_job.py` to change the business ID and email recipient.

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Make sure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: churn_doctor_data/"
**Solution:** Run data generation first:
```bash
python generate_data.py
```

### Issue: "FileNotFoundError: models/churn_model.pkl"
**Solution:** Train the model first:
```bash
python churnmodel.py
```

### Issue: Models downloading slowly
**Solution:** This is normal on first run. Models are cached locally after first download.

### Issue: Out of memory errors
**Solution:** The models are large. Close other applications or use a machine with more RAM (8GB+ recommended).

---

## Project Structure

```
AML - CHURN DOCTOR/
├── churnmodel.py          # Train the churn prediction model
├── dashboard.py           # Streamlit dashboard (main UI)
├── features.py            # Feature engineering
├── explainer.py          # SHAP explanations
├── llm.py                 # Free LLM (Hugging Face) - NO API KEY NEEDED
├── insight_engine.py      # Generate insights
├── emailer.py             # Email notifications (optional)
├── schedular_job.py       # Scheduled analysis job
├── generate_data.py       # Generate sample data
├── config.py              # Configuration (no API keys needed)
├── requirements.txt       # Python dependencies
└── churn_doctor_data/     # Data directory (created after step 3)
    ├── businesses.csv
    ├── customers.csv
    ├── orders.csv
    └── interactions.csv
└── models/                 # Model directory (created after step 4)
    └── churn_model.pkl
```

---

## Summary

✅ **No API keys needed** - All models run locally for free  
✅ **Simple setup** - Just install dependencies and run  
✅ **Fast dashboard** - Streamlit web interface  
✅ **Optional email** - Configure only if you want email reports  

Enjoy using Churn Doctor! 🎉

