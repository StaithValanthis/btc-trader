from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self):
        """
        Initialize the machine learning model.
        """
        self.model = self.build_model()
        self.is_trained = False  # Track whether the model has been trained
        self.feature_shape = None  # Store the expected input feature shape

    def build_model(self):
        """
        Build the machine learning pipeline.
        
        Returns:
            Pipeline: A scikit-learn pipeline with preprocessing and classifier.
        """
        return Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('classifier', RandomForestClassifier(
                n_estimators=100,  # Number of trees in the forest
                max_depth=5,  # Maximum depth of each tree
                random_state=42,  # Seed for reproducibility
                class_weight='balanced_subsample',  # Handle class imbalance
                verbose=1,  # Print training progress
                n_jobs=-1  # Use all available CPU cores
            ))
        ])

    def train(self, X, y):
        """
        Train the model on the provided data.
        
        Args:
            X (numpy.ndarray): Training features.
            y (numpy.ndarray): Training labels.
            
        Returns:
            bool: True if training succeeded, False otherwise.
        """
        try:
            # Validate input data
            if X.shape[0] < 50:
                raise ValueError("Minimum 50 samples required for training")
                
            # Store feature shape for validation during prediction
            self.feature_shape = X.shape[1]
            
            # Train the model
            self.model.fit(X, y)
            self.is_trained = True
            logger.info(f"Model trained successfully on {X.shape[0]} samples")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.is_trained = False
            return False

    def predict(self, X):
        """
        Generate predictions using the trained model.
        
        Args:
            X (numpy.ndarray): Input features for prediction.
            
        Returns:
            numpy.ndarray: Predicted probabilities for the positive class.
            
        Raises:
            NotFittedError: If the model is not trained.
            ValueError: If the input feature shape is incorrect.
        """
        # Validate model state
        if not self.is_trained:
            raise NotFittedError("Model must be trained before making predictions")
            
        # Validate input shape
        if X.shape[1] != self.feature_shape:
            raise ValueError(f"Expected {self.feature_shape} features, got {X.shape[1]}")
            
        try:
            # Generate predictions
            return self.model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return np.zeros(X.shape[0])  # Return neutral predictions on error