#!/usr/bin/env python3
"""
Script to set up the Excel file for the MIT AI Risk Repository Chatbot.
This script checks if the Excel file exists in the current directory or Downloads
and copies it to the info_files directory.
"""

import os
import shutil
from pathlib import Path
import sys

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    info_files_dir = os.path.join(current_dir, 'info_files')
    excel_filename = "The_AI_Risk_Repository_V3_26_03_2025.xlsx"
    
    # Make sure info_files directory exists
    os.makedirs(info_files_dir, exist_ok=True)
    
    # Target path
    target_path = os.path.join(info_files_dir, excel_filename)
    
    # If file already exists in info_files, we're done
    if os.path.exists(target_path):
        print(f"✅ Excel file already exists at: {target_path}")
        return True
    
    # Check common locations
    potential_locations = [
        current_dir,                                       # Current directory
        os.path.join(os.path.expanduser("~"), "Downloads"),  # Downloads folder
        os.path.join(os.path.expanduser("~"), "Desktop"),    # Desktop folder
        os.path.join(os.path.expanduser("~"), "Documents"),  # Documents folder
    ]
    
    # Try to find the file
    excel_path = None
    for location in potential_locations:
        path = os.path.join(location, excel_filename)
        if os.path.exists(path):
            excel_path = path
            break
    
    # If we didn't find it, ask the user
    if not excel_path:
        print(f"❌ Could not find {excel_filename} in common locations.")
        user_path = input(f"Please enter the full path to {excel_filename}: ")
        if os.path.exists(user_path):
            excel_path = user_path
        else:
            print(f"❌ File not found at: {user_path}")
            return False
    
    # Copy the file
    try:
        shutil.copy2(excel_path, target_path)
        print(f"✅ Excel file copied to: {target_path}")
        return True
    except Exception as e:
        print(f"❌ Error copying file: {str(e)}")
        return False

if __name__ == "__main__":
    result = main()
    if not result:
        sys.exit(1)