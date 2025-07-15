import os

class Config:
    """Base configuration class."""
    DEBUG = True

class DevelopmentConfig(Config):
    """Development configuration class."""
    DEBUG = False

class ProductionConfig(Config):
    """Production configuration class."""
    DEBUG = True