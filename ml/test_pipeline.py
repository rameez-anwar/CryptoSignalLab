#!/usr/bin/env python3
"""
Test script for ML pipeline
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test that all modules can be imported correctly"""
    print("Testing imports...")
    
    try:
        from learner.base_learner import BaseLearner
        print("✓ BaseLearner imported successfully")
        
        # Test base learner initialization
        base_learner = BaseLearner()
        available_models = base_learner.get_available_models()
        print(f"✓ Available models: {available_models}")
        
        # Test that we can get parameters for each model
        for model_name in available_models:
            try:
                params = base_learner.get_model_params(model_name)
                param_ranges = base_learner.get_model_param_ranges(model_name)
                print(f"✓ {model_name}: params={len(params)} keys, ranges={len(param_ranges)} keys")
            except Exception as e:
                print(f"✗ {model_name}: {str(e)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {str(e)}")
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from main import load_config
        config = load_config("config.ini")  # Use relative path
        print(f"✓ Config loaded successfully")
        print(f"  Exchange: {config['data']['exchange']}")
        print(f"  Symbol: {config['data']['symbol']}")
        print(f"  Time Horizon: {config['data']['time_horizon']}")
        print(f"  Models: {config['models']['selected_models']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config error: {str(e)}")
        return False

def test_sample_data():
    """Test sample data generation"""
    print("\nTesting sample data generation...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Generate sample data
        start_dt = pd.to_datetime('2024-01-01')
        end_dt = pd.to_datetime('2024-01-31')
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq='4h')  # Use lowercase 'h'
        
        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0, 0.02, len(timestamps))
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            volatility = 0.01
            open_price = price * (1 + np.random.normal(0, volatility/2))
            high_price = max(open_price, price) * (1 + abs(np.random.normal(0, volatility/4)))
            low_price = min(open_price, price) * (1 - abs(np.random.normal(0, volatility/4)))
            close_price = price
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(1000, 10000)
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        print(f"✓ Sample data generated: {len(df)} data points")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except Exception as e:
        print(f"✗ Sample data error: {str(e)}")
        return None

def test_model_training():
    """Test model training with sample data"""
    print("\nTesting model training...")
    
    try:
        from learner.base_learner import BaseLearner
        import numpy as np
        
        # Generate sample data
        data = test_sample_data()
        if data is None:
            return False
        
        # Initialize base learner
        base_learner = BaseLearner()
        
        # Test training with one model
        model_name = 'linear_regression_model'
        params = base_learner.get_model_params(model_name)
        
        print(f"Training {model_name}...")
        val_loss = base_learner.train_model(model_name, data, params)
        print(f"✓ {model_name} trained successfully, validation loss: {val_loss:.6f}")
        
        # Test signal generation
        signals = base_learner.generate_signals(model_name, data, params)
        print(f"✓ Signals generated: {len(signals)} signals")
        
        # Count signal distribution
        unique, counts = np.unique(signals, return_counts=True)
        signal_dist = dict(zip(unique, counts))
        print(f"  Signal distribution: {signal_dist}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== ML Pipeline Test Suite ===")
    
    tests = [
        test_imports,
        test_config,
        test_model_training
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 