"""
Configuration Module

Central configuration file for the AI Travel Recommendation System.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration."""
    
    # Flask settings
    DEBUG = False
    TESTING = False
    
    # Application settings
    APP_NAME = 'AI Travel Recommendation System'
    APP_VERSION = '1.0.0'
    
    # Data paths
    DATA_RAW_PATH = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    DATA_PROCESSED_PATH = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    
    # Model paths
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
    
    # Database settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')


class DevelopmentConfig(Config):
    """Development configuration."""
    
    DEBUG = True


class TestingConfig(Config):
    """Testing configuration."""
    
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'


class ProductionConfig(Config):
    """Production configuration."""
    
    DEBUG = False


def get_config(env: str = None) -> Config:
    """
    Get configuration based on environment.
    
    Args:
        env: Environment name ('development', 'testing', 'production')
        
    Returns:
        Configuration object
    """
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    
    if env == 'testing':
        return TestingConfig()
    elif env == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig()
