# FIFA Players Analysis

A machine learning project for analyzing FIFA players data and predicting player positions.

## Features

- Exploratory Data Analysis (EDA) with interactive visualizations
- Feature engineering and preprocessing
- Model training with hyperparameter optimization using Optuna
- Prediction pipeline for new data
- Command-line interface for easy usage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FIFA_players_analysis.git
cd FIFA_players_analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project provides a command-line interface with three main commands:

1. Generate EDA reports:
```bash
python cli.py eda fifa_players.csv reports/
```

2. Train a model:
```bash
python cli.py train fifa_players.csv --target player_positions
```

3. Make predictions:
```bash
python cli.py predict model.pkl new.csv preds.csv
```

To see all available options and examples:
```bash
python cli.py --examples
```

## Project Structure

- `cli.py` - Main entry point and command-line interface
- `data.py` - Data loading and preprocessing utilities
- `features.py` - Feature engineering functions
- `model.py` - Model training and evaluation code
- `requirements.txt` - Project dependencies
- `reports/` - Directory for generated EDA reports

## License

MIT License 