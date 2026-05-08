#!/usr/bin/env python3
"""
Standalone script to train the predictive maintenance model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime
from src.model_trainer import ModelTrainer
from src.data_loader import DataLoader
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train predictive maintenance model')
    parser.add_argument('--data', type=str, default='simulated', 
                       help='Data source: simulated or path to CSV')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of samples for simulated data')
    parser.add_argument('--models', nargs='+', default=['rf', 'gb', 'svm'],
                       help='Models to train: rf (Random Forest), gb (Gradient Boosting), svm (SVM)')
    parser.add_argument('--output', type=str, default='models/best_model.pkl',
                       help='Output model path')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size ratio')
    
    args = parser.parse_args()
    
    logger.info(f"Starting model training with arguments: {args}")
    
    try:
        # Load data
        data_loader = DataLoader()
        
        if args.data == 'simulated':
            logger.info(f"Generating {args.samples} simulated samples...")
            df = data_loader.load_simulated_data(args.samples)
        else:
            logger.info(f"Loading data from {args.data}...")
            df = data_loader.load_csv(args.data)
            
        if df is None or df.empty:
            logger.error("No data loaded. Exiting.")
            sys.exit(1)
        
        # Train model
        trainer = ModelTrainer()
        
        logger.info("Training models...")
        results = trainer.train_models(
            df=df,
            model_types=args.models,
            test_size=args.test_size
        )
        
        # Save best model
        if trainer.best_model:
            trainer.save_model(args.output)
            logger.info(f"Best model saved to {args.output}")
            
            # Save metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'best_model': trainer.best_model_name,
                'best_accuracy': trainer.best_accuracy,
                'models_trained': list(results.keys()),
                'parameters': vars(args)
            }
            
            with open('models/training_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Training metadata saved")
            
            # Print results
            print("\n" + "="*60)
            print("MODEL TRAINING RESULTS")
            print("="*60)
            for model_name, result in results.items():
                print(f"\n{model_name}:")
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  Precision: {result['precision']:.4f}")
                print(f"  Recall: {result['recall']:.4f}")
                print(f"  F1-Score: {result['f1']:.4f}")
            
            print(f"\nBest Model: {trainer.best_model_name}")
            print(f"Best Accuracy: {trainer.best_accuracy:.4f}")
            print("="*60)
            
        else:
            logger.error("No model was trained successfully")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()