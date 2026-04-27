"""
Main entry point for Azure deployment
This file should be at the root level for Azure to detect it
"""
import os
import sys

# Add web_app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_app'))

# Import the Flask app from web_app/app.py
from web_app.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
