"""
Main entry point for Azure deployment
"""
import os
import sys

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add web_app directory to Python path
web_app_dir = os.path.join(current_dir, 'web_app')
sys.path.insert(0, web_app_dir)

# Add src directory to Python path (for model imports)
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import Flask app
try:
    os.chdir(web_app_dir)
    from app import app
    
    if __name__ == '__main__':
        # Get port from environment (Azure sets this)
        port = int(os.environ.get('PORT', 8000))
        print(f"Starting app on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
except Exception as e:
    print(f"Error starting app: {e}")
    import traceback
    traceback.print_exc()
    raise
