import os
import subprocess
import sys

def run_streamlit():
    # Get the current directory
    current_dir = os.getcwd()
    
    # Set the Streamlit server port
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    
    # Set the Streamlit server address
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    
    # Set the Streamlit browser server address
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Run Streamlit
    subprocess.run([
        sys.executable, 
        '-m', 
        'streamlit', 
        'run', 
        os.path.join(current_dir, 'app.py'),
        '--server.port=8501',
        '--server.address=0.0.0.0',
        '--browser.serverAddress=0.0.0.0',
        '--browser.serverPort=8501'
    ])

if __name__ == "__main__":
    run_streamlit() 