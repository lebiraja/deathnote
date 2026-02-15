# Life Expectancy Prediction Model 🏥

A comprehensive machine learning project to predict life expectancy based on health and lifestyle factors using a modern, animated Flask web application.

## 🎯 Key Features

- **AI-Powered Predictions**: Uses Gradient Boosting model with 87% accuracy
- **Modern Web UI**: Beautiful, animated Flask interface with real-time interactions
- **Comprehensive Analysis**: Analyzes 14 health factors
- **Personalized Insights**: Custom health recommendations based on your profile
- **Report Generation**: Download your prediction report
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## Project Structure

```
deathnote/
├── life-expectancy.csv          # Dataset
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── data_loader.py               # Module for loading data
├── eda.py                        # Exploratory Data Analysis module
├── preprocessing.py             # Data preprocessing module
├── model.py                      # Model training and management
├── train.py                      # Training pipeline script
├── app.py                        # Streamlit prediction app
│
├── models/                       # Directory for saved artifacts
│   ├── linear_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── scaler.pkl
│   └── preprocessor.pkl
│
└── eda_outputs/                  # Directory for EDA visualizations
    ├── 01_target_distribution.png
    ├── 02_categorical_analysis.png
    ├── 03_correlation_heatmap.png
    └── 04_numerical_scatter.png
```

## Features

### Input Features
- **Personal Information**: Gender, Height, Weight, BMI
- **Lifestyle Factors**: Physical Activity, Smoking Status, Alcohol Consumption, Diet
- **Health Indicators**: Blood Pressure, Cholesterol
- **Medical History**: Diabetes, Hypertension, Heart Disease, Asthma

### Target Variable
- **Age / Life Expectancy**: Years of life expectancy

## Installation

1. **Clone/Navigate to the project directory:**
   ```bash
   cd deathnote
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Model

Run the training pipeline to perform EDA, preprocess data, and train models:

```bash
python train.py
```

This will:
- Load and analyze the dataset
- Generate EDA visualizations in `eda_outputs/`
- Preprocess features and split data
- Train three models: Linear Regression, Random Forest, and Gradient Boosting
- Save the best model and preprocessing artifacts
- Display performance metrics

### Step 2: Run the Flask Web Application

Once training is complete, start the Flask app with the modern animated UI:

```bash
python flask_app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

The web application features:
- ✨ Beautiful, modern animated interface
- 🎯 Interactive form with real-time BMI calculation
- 📊 Instant life expectancy predictions
- 💡 Personalized health insights
- 🎁 Custom health recommendations
- 📥 Download prediction reports
- 📱 Fully responsive design

### Alternative: Run the Streamlit App (Legacy)

If you prefer the Streamlit interface:

```bash
streamlit run app.py
```

## Models

The project trains and compares three regression models:

1. **Linear Regression**: Simple, interpretable baseline
2. **Random Forest**: Ensemble method with good generalization
3. **Gradient Boosting**: Advanced boosting method for better performance

Each model is evaluated on Train, Validation, and Test sets using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Modules Overview

### `data_loader.py`
- `load_data()`: Load the CSV dataset
- `get_dataset_info()`: Display dataset statistics

### `eda.py`
- `EDA` class: Performs comprehensive exploratory data analysis
- Generates visualizations for distributions, correlations, and relationships
- Saves plots to `eda_outputs/` directory

### `preprocessing.py`
- `DataPreprocessor` class: Handles all preprocessing tasks
  - Encodes categorical features using LabelEncoder
  - Splits data into train/val/test sets
  - Scales features using StandardScaler
  - Saves/loads preprocessing artifacts

### `model.py`
- `LifeExpectancyModel` class: Trains and manages models
  - Supports multiple model types
  - Provides training and evaluation
  - Saves/loads models and scalers
  - Makes predictions on new data

### `train.py`
- Complete training pipeline
- Orchestrates all steps from data loading to model training
- Compares all models and selects the best one

### `app.py`
- Streamlit web application
- Interactive user interface for predictions
- Health recommendations based on profile
- Beautiful visualizations

## Dataset

The dataset contains 10,002 records with 15 features:
- 1 Target variable (Age/Life Expectancy)
- 14 Input features (categorical and numerical)

**Categorical features**: Gender, Physical_Activity, Smoking_Status, Alcohol_Consumption, Diet, Blood_Pressure
**Numerical features**: Height, Weight, BMI, Cholesterol, Diabetes, Hypertension, Heart_Disease, Asthma

## Performance Metrics

After training, the models will display metrics like:
```
Test R² Score: 0.85-0.92
Test RMSE: 3-5 years (depending on model)
```

## Tips for Better Predictions

1. **Accurate Input**: Provide truthful health information for better predictions
2. **Regular Checkups**: Keep medical records updated
3. **Lifestyle Changes**: The model can show impact of lifestyle improvements
4. **Professional Advice**: Always consult healthcare providers for medical decisions

## Future Enhancements

- Add more health indicators
- Implement cross-validation
- Add hyperparameter tuning
- Create confidence intervals for predictions
- Add data drift monitoring
- Deploy as a REST API

## License

This project is for educational purposes.

## Support

For issues or questions, please check the code documentation or modify the modules as needed.


CURRENTLY THE SERVICE IS DOWN!!!!!!!!!!!!!!
