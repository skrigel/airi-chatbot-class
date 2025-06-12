#!/usr/bin/env python3
import os
import subprocess
import sys

print("Starting setup process...")

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
setup_script = os.path.join(current_dir, "setup.sh")

# Make the script executable
try:
    os.chmod(setup_script, 0o755)
    print("Made setup.sh executable")
except Exception as e:
    print(f"Error making setup.sh executable: {str(e)}")

# Run the setup script
try:
    print("Running setup.sh...")
    process = subprocess.Popen([setup_script], 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True,
                              bufsize=1)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for process to complete
    process.wait()
    
    # Get return code
    return_code = process.returncode
    
    if return_code == 0:
        print("Setup completed successfully!")
    else:
        print(f"Setup failed with return code {return_code}")
        for line in process.stderr:
            print(line, end='')
except Exception as e:
    print(f"Error running setup script: {str(e)}")

print("Setup process finished.")