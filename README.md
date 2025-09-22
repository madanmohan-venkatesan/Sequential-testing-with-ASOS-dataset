# A/B Testing Sequential Analysis System

A comprehensive system for sequential A/B testing analysis using STF and SAVVY libraries, built with FastAPI and Streamlit.

## Features

- **Sequential Testing**: Uses STF for continuous metrics and SAVVY for binary/count metrics
- **Sample Ratio Mismatch (SRM) Detection**: Automatically detects SRM issues
- **Real-time Visualization**: Interactive plots showing p-value history and distributions
- **REST API**: FastAPI backend for programmatic access
- **Web Interface**: Streamlit frontend for interactive analysis
- **Modular Design**: SOLID principles with clean separation of concerns

## Architecture

The system is built with the following components:

- **Data Collection**: Downloads and manages ASOS dataset
- **Data Preprocessing**: Validates and converts data types
- **Data Generation**: Generates synthetic data for count-based metrics
- **Data Filtering**: Extracts specific experiment data
- **Inference Engine**: Performs sequential testing with STF/SAVVY
- **Visualization**: Creates comprehensive plots and reports
- **API Backend**: FastAPI REST API
- **Web Frontend**: Streamlit user interface

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sequential-test-using-asos-dataset
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\Activate.ps1
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install uv
   uv sync
   ```

## Usage

### 1. Start the FastAPI Backend

```bash
python src/api.py
```

The API will be available at `http://localhost:8000`

### 2. Start the Streamlit Frontend

```bash
streamlit run src/streamlit_app.py
```

The web interface will be available at `http://localhost:8501`

### 3. API Endpoints

- `GET /health` - Health check
- `GET /experiments/available` - List available experiments
- `POST /experiments/analyze` - Run sequential analysis
- `GET /experiments/{id}/variants/{id}/metrics/{id}/summary` - Get experiment summary
- `GET /plots/{experiment_id}/variant{variant_id}/metric{metric_id}/{plot_type}` - Get plots

### 4. Programmatic Usage

```python
from src.main import run_analysis

# Run complete analysis
results = run_analysis(
    experiment_id="036afc",
    variant_id=2,
    metric_id=1,
    hypothesis="a=b"
)

print(f"Significance detected: {results['results']['significance_detected']}")
print(f"Final p-value: {results['results']['final_p_value']}")
```

## Dataset

The system uses the ASOS Digital Experiments Dataset, which includes:

- **experiment_id**: Anonymized experiment identifier
- **variant_id**: Treatment group identifier
- **metric_id**: Metric identifier (1=binary, 2=count, 3=count, 4=continuous)
- **time_since_start**: Days since experiment start
- **count_c/mean_c/variance_c**: Control group statistics
- **count_t/mean_t/variance_t**: Treatment group statistics

## Sequential Testing

The system performs sequential testing with the following logic:

1. **Metric Type Detection**: Determines if metric is binary, count, or continuous
2. **Sequential Testing**: 
   - Continuous metrics: Uses STF library
   - Binary/Count metrics: Uses SAVVY library
3. **SRM Testing**: Checks for sample ratio mismatch using chi-square test
4. **Early Stopping**: Stops when significance is detected
5. **Traditional Comparison**: Compares with traditional A/B test results

## Visualization

The system creates comprehensive visualizations:

- **Distribution Analysis**: Control vs treatment distributions over time
- **P-value History**: Sequential test p-value progression
- **SRM History**: Sample ratio mismatch p-value progression
- **Final Comparison**: Sequential vs traditional test results
- **Comprehensive Report**: All analyses in one view

## Configuration

Key parameters can be configured:

- **Alpha (α)**: Type I error rate (default: 0.05)
- **Beta (β)**: Type II error rate (default: 0.2)
- **Hypothesis**: Test direction ('a>=b', 'a<=b', 'a=b')
- **Visualization**: Enable/disable plot generation

## Development

### Project Structure

```
src/
├── data_collection.py      # Dataset download and loading
├── data_preprocessing.py   # Data validation and type conversion
├── data_generation.py       # Synthetic data generation
├── data_filtering.py       # Experiment data extraction
├── inference_engine.py     # Sequential testing logic
├── visualization.py        # Plotting and visualization
├── api.py                 # FastAPI backend
├── streamlit_app.py       # Streamlit frontend
└── main.py               # Main orchestration module

data/                      # Dataset storage
figures/                   # Generated plots
```

### Testing

```bash
# Run tests
pytest tests/

# Run specific module tests
python src/data_collection.py
python src/data_preprocessing.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- ASOS for providing the experimental dataset
- STF and SAVVY libraries for sequential testing capabilities
- FastAPI and Streamlit communities for excellent frameworks
