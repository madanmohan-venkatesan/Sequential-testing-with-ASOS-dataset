#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data_collection import DataCollector
        print("✅ data_collection imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data_collection: {e}")
        return False
    
    try:
        from data_preprocessing import DataPreprocessor
        print("✅ data_preprocessing imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data_preprocessing: {e}")
        return False
    
    try:
        from data_generation import DataGenerator
        print("✅ data_generation imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data_generation: {e}")
        return False
    
    try:
        from data_filtering import DataFilter
        print("✅ data_filtering imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data_filtering: {e}")
        return False
    
    try:
        from inference_engine import SequentialTester
        print("✅ inference_engine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import inference_engine: {e}")
        return False
    
    try:
        from visualization import ExperimentVisualizer
        print("✅ visualization imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import visualization: {e}")
        return False
    
    try:
        from main import ABTestOrchestrator
        print("✅ main imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import main: {e}")
        return False
    
    return True

def test_data_collection():
    """Test data collection functionality."""
    print("\nTesting data collection...")
    
    try:
        from data_collection import DataCollector
        collector = DataCollector()
        
        # Test directory creation
        print("✅ DataCollector initialized successfully")
        
        # Test dataset download (this might take a while)
        print("Downloading dataset (this may take a few minutes)...")
        df = collector.get_dataset()
        
        if df is not None:
            print(f"✅ Dataset loaded successfully. Shape: {df.shape}")
            return True
        else:
            print("❌ Failed to load dataset")
            return False
            
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("\nTesting data preprocessing...")
    
    try:
        from data_collection import DataCollector
        from data_preprocessing import DataPreprocessor
        
        # Load data
        collector = DataCollector()
        df = collector.get_dataset()
        
        if df is None:
            print("❌ Cannot test preprocessing without data")
            return False
        
        # Test preprocessing
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess_data(df)
        
        print(f"✅ Data preprocessing successful. Shape: {df_processed.shape}")
        print(f"✅ Data types: {df_processed.dtypes.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        return False

def test_available_experiments():
    """Test getting available experiments."""
    print("\nTesting available experiments...")
    
    try:
        from main import ABTestOrchestrator
        
        orchestrator = ABTestOrchestrator()
        experiments = orchestrator.get_available_experiments()
        
        print(f"✅ Found {len(experiments)} available experiments")
        
        if len(experiments) > 0:
            first_exp = experiments[0]
            print(f"✅ First experiment: {first_exp['experiment_id']}, "
                  f"variant {first_exp['variant_id']}, metric {first_exp['metric_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Available experiments test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running installation tests...\n")
    
    tests = [
        test_imports,
        test_data_collection,
        test_data_preprocessing,
        test_available_experiments
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("1. Start the API: python start_api.py")
        print("2. Start Streamlit: python start_streamlit.py")
        print("3. Open http://localhost:8501 in your browser")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
