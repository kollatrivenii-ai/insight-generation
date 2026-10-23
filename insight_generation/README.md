# Insight Generation

Automatically generates ranked, plain-English insights from a CSV dataset
(missing values, outliers, skew, correlations, dominant categories,
group differences, and time trends).

## Setup
    pip install -r requirements.txt

## Run
    python insight_generator.py sample_data.csv   # included sample
    python insight_generator.py your_data.csv     # your own CSV
    python insight_generator.py                   # built-in demo data

## Extend
Add a function that returns a list of (score, text) pairs and register it
in `generate_insights()`.
