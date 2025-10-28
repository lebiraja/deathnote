"""
Quick start guide for the Life Expectancy Prediction project
"""

# Quick Start Guide ⚡

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Train the Model

```bash
python train.py
```

Expected output:
- Loads 10,002 records from life-expectancy.csv
- Generates EDA visualizations
- Trains 3 models (Linear, Random Forest, Gradient Boosting)
- Saves best model to models/ directory
- Takes 2-5 minutes depending on your machine

## 3. Run the Prediction App

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## File Descriptions

| File | Purpose |
|------|---------|
| `data_loader.py` | Load and inspect data |
| `eda.py` | Exploratory Data Analysis |
| `preprocessing.py` | Feature encoding and scaling |
| `model.py` | Model training and prediction |
| `train.py` | Complete training pipeline |
| `app.py` | Interactive Streamlit application |

## Expected Results

After training, you should see:
- Random Forest typically achieves R² > 0.85 on test set
- RMSE around 3-5 years
- All models save to `models/` directory
- EDA plots save to `eda_outputs/` directory

## Troubleshooting

**ImportError for pandas/sklearn?**
- Run: `pip install -r requirements.txt`

**Model not found when running app.py?**
- Run `python train.py` first to train and save models

**Streamlit app won't start?**
- Ensure all dependencies are installed
- Check port 8501 is available

## Next Steps

1. Experiment with the app interface
2. Review EDA plots in `eda_outputs/`
3. Check model performance in the terminal output
4. Try different health profiles to see predictions
5. Modify models in `model.py` for experimentation
