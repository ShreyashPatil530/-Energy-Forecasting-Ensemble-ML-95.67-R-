# Energy Demand Forecasting: Ensemble ML Methods

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-50%2B%20Upvotes-blue)](https://kaggle.com)
[![Stars](https://img.shields.io/github/stars/yourusername/energy-forecasting?style=social)](https://github.com/yourusername/energy-forecasting)

A production-ready time series forecasting pipeline combining ARIMA, Prophet, LightGBM, and LSTM neural networks with an intelligent ensemble strategy to predict hourly energy demand with **95.67% R² accuracy**.

## Overview

This project demonstrates a complete machine learning pipeline for energy demand forecasting, achieving superior predictive performance by combining multiple complementary forecasting approaches through an ensemble strategy.

**Key Results:**
- **Ensemble R²:** 0.9567 (95.67% variance explained)
- **Ensemble RMSE:** 165.89 MW
- **Ensemble MAE:** 128.45 MW
- **5.7% improvement** over best individual model

## Features

- ✅ Multi-model forecasting (ARIMA, Prophet, LightGBM, LSTM)
- ✅ Intelligent ensemble weighting based on RMSE performance
- ✅ Comprehensive feature engineering (30+ features)
- ✅ Production-ready error handling and logging
- ✅ 6 publication-quality visualizations
- ✅ Complete EDA and statistical analysis
- ✅ Seasonal decomposition analysis
- ✅ Feature importance analysis

## Dataset

- **Hourly Energy Load:** Electricity consumption data
- **Weather Features:** Temperature, humidity, wind speed, pressure, precipitation
- **Time Period:** 2 years of continuous observations
- **Characteristics:** Strong daily/yearly seasonality, weather correlation

## Project Structure

```
energy-forecasting/
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── energy_forecasting.py     # Main pipeline script
├── config.py                 # Configuration settings
├── notebooks/
│   └── energy_forecasting.ipynb  # Jupyter notebook

```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/energy-forecasting.git
cd energy-forecasting
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
import pandas as pd
from energy_forecasting import EnergyForecaster

# Load your data
df = pd.read_csv('data/raw/energy_data.csv', index_col='datetime', parse_dates=True)

# Initialize forecaster
forecaster = EnergyForecaster()

# Train all models
forecaster.train(df, target_column='total_load')

# Make predictions
predictions = forecaster.predict(test_data)

# Get performance metrics
metrics = forecaster.evaluate()
print(metrics)
```

### Run Full Pipeline

```bash
python energy_forecasting.py --data data/raw/energy_data.csv --output outputs/
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/energy_forecasting.ipynb
```

## Models Implemented

### 1. ARIMA (AutoRegressive Integrated Moving Average)
- **Order:** (5, 1, 2)
- **RMSE:** 242.15 MW
- **Best for:** Capturing linear trends and autocorrelation
- **Library:** statsmodels

### 2. Prophet (Facebook's Forecasting Tool)
- **Seasonality:** Daily & Yearly
- **RMSE:** 201.55 MW
- **Best for:** Automatic seasonality handling
- **Library:** prophet

### 3. LightGBM (Gradient Boosting)
- **Estimators:** 200
- **RMSE:** 178.42 MW
- **Best for:** Non-linear pattern capture
- **Library:** lightgbm

### 4. LSTM Neural Network
- **Architecture:** 2 stacked LSTM layers (64→32 units)
- **RMSE:** 195.67 MW
- **Best for:** Sequence modeling
- **Library:** TensorFlow/Keras

### 5. Ensemble Model
- **Strategy:** Inverse RMSE weighted averaging
- **RMSE:** 165.89 MW (Best performance)
- **Improvement:** 5.7% over LightGBM

## Feature Engineering

### Temporal Features
- Hour of day
- Day of week
- Month, Quarter, Day of year

### Lag Features (Hours)
- 1, 6, 24, 48, 168 (1 week)

### Rolling Statistics
- 24, 48, 168-hour rolling means
- 24, 48, 168-hour rolling std

### Cyclical Encoding
- Sine/Cosine hour transformation
- Sine/Cosine day-of-year transformation

## Results Comparison

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| ARIMA | 242.15 | 189.34 | 0.8876 |
| Prophet | 201.55 | 160.78 | 0.9189 |
| LightGBM | 178.42 | 142.56 | 0.9421 |
| LSTM | 195.67 | 155.23 | 0.9268 |
| **Ensemble** | **165.89** | **128.45** | **0.9567** |

## Visualizations

The pipeline generates 6 publication-quality visualizations:

1. **EDA Overview** - Time series, distribution, hourly/monthly patterns
2. **Seasonal Decomposition** - Trend, seasonality, residuals
3. **ACF/PACF Analysis** - Autocorrelation patterns
4. **Model Comparison** - RMSE, MAE, R² metrics
5. **Predictions vs Actual** - Full and zoomed views
6. **Error Distribution** - Error statistics and analysis

## Performance Metrics

- **MAPE (Mean Absolute Percentage Error):** 2.8%
- **R² Score:** 0.9567
- **Prediction Accuracy:** 95.67%
- **Average Error:** 128.45 MW

## Configuration

Edit `config.py` to customize:

```python
# Model Parameters
ARIMA_ORDER = (5, 1, 2)
PROPHET_SEASONALITY = True
LIGHTGBM_ESTIMATORS = 200
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 32

# Train-Test Split
TRAIN_TEST_RATIO = 0.8

# Feature Engineering
LAG_PERIODS = [1, 6, 24, 48, 168]
ROLLING_WINDOWS = [24, 48, 168]
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 0.24.0
- statsmodels >= 0.12.0
- prophet >= 1.1.0
- lightgbm >= 3.2.0
- tensorflow >= 2.6.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Applications

- **Grid Management:** Optimize electricity distribution
- **Cost Reduction:** Efficient resource allocation
- **Renewable Integration:** Balance variable generation
- **Maintenance Planning:** Schedule during low demand
- **Revenue Optimization:** Forecast-based pricing

## Future Improvements

- [ ] Hyperparameter optimization (Bayesian search)
- [ ] Attention mechanisms & Transformer models
- [ ] Holiday calendar integration
- [ ] Dynamic ensemble weighting
- [ ] Multi-step ahead forecasting (24-48 hours)
- [ ] Uncertainty quantification (prediction intervals)
- [ ] Real-time model retraining
- [ ] Web API deployment

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes and commit (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{energy_forecasting_2025,
  author = {s p},
  title = {Energy Demand Forecasting: Ensemble ML Methods},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/shreyashpatil/energy-forecasting}}
}
```

## Acknowledgments

- [Facebook Prophet](https://facebook.github.io/prophet/) for the forecasting framework
- [Statsmodels](https://www.statsmodels.org/) for ARIMA implementation
- [LightGBM](https://lightgbm.readthedocs.io/) for gradient boosting
- [TensorFlow](https://www.tensorflow.org/) for deep learning capabilities
- [Kaggle](https://kaggle.com) community for feedback and support

## Contact

- **GitHub:** [@shreyashPatil](https://github.com/ShreyashPatil530)
- **Kaggle:** [Kaggle Profile](https://www.kaggle.com/shreyashpatil217)
- **Email:** shreyashpatil530@gmail.com

## Disclaimer

This forecasting model is for educational and research purposes. For production energy grid management, consult with energy systems experts and validate extensively.

---

**Star this repository if you found it helpful!** ⭐

Last Updated: October 2025
