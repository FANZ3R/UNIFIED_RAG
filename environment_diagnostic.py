"""
Environment diagnostic script to identify Python environment issues
"""

import sys
import os
import subprocess

def check_current_environment():
    """Check current Python environment details"""
    print("🔍 CURRENT PYTHON ENVIRONMENT")
    print("=" * 50)
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[0]}")
    
    # Check if we're in a conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"Conda environment: {conda_env}")
    else:
        print("Not in a conda environment")
    
    # Check virtual environment
    virtual_env = os.environ.get('VIRTUAL_ENV')
    if virtual_env:
        print(f"Virtual environment: {virtual_env}")
    else:
        print("Not in a virtual environment")

def check_package_locations():
    """Check where packages are installed"""
    print(f"\n📦 PACKAGE LOCATIONS")
    print("=" * 50)
    
    packages_to_check = [
        'streamlit',
        'qdrant_client', 
        'sentence_transformers',
        'openai',
        'neo4j'
    ]
    
    for package in packages_to_check:
        try:
            exec(f"import {package}")
            module = sys.modules[package]
            location = getattr(module, '__file__', 'Location unknown')
            if location:
                # Get the site-packages directory
                site_packages = location.split('site-packages')[0] + 'site-packages' if 'site-packages' in location else location
                print(f"✅ {package}: {site_packages}")
            else:
                print(f"✅ {package}: Built-in module")
        except ImportError as e:
            print(f"❌ {package}: Not found - {e}")

def check_streamlit_command():
    """Check which streamlit command is being used"""
    print(f"\n🚀 STREAMLIT COMMAND CHECK")
    print("=" * 50)
    
    try:
        # Check which streamlit
        result = subprocess.run(['which', 'streamlit'], capture_output=True, text=True)
        if result.returncode == 0:
            streamlit_path = result.stdout.strip()
            print(f"Streamlit command location: {streamlit_path}")
            
            # Check if it's in the same environment
            current_python_dir = os.path.dirname(sys.executable)
            streamlit_dir = os.path.dirname(streamlit_path)
            
            if current_python_dir == streamlit_dir:
                print("✅ Streamlit is in the same environment as current Python")
            else:
                print("❌ Streamlit is in a DIFFERENT environment than current Python!")
                print(f"   Current Python: {current_python_dir}")
                print(f"   Streamlit: {streamlit_dir}")
        else:
            print("❌ Streamlit command not found in PATH")
    
    except Exception as e:
        print(f"Error checking streamlit command: {e}")

def suggest_fixes():
    """Suggest fixes based on the diagnosis"""
    print(f"\n🔧 SUGGESTED FIXES")
    print("=" * 50)
    
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    
    if conda_env and conda_env == 'unified_rag':
        print("✅ You are in the unified_rag conda environment")
        print("\n🔧 Try these commands:")
        print("1. Reinstall packages in current environment:")
        print("   pip uninstall qdrant-client sentence-transformers")
        print("   pip install qdrant-client sentence-transformers")
        print("\n2. Or use conda to install:")
        print("   conda install -c conda-forge sentence-transformers")
        print("   pip install qdrant-client  # (not available in conda)")
        print("\n3. Run streamlit with full path:")
        print(f"   {sys.executable} -m streamlit run unified_chatbot.py")
        
    else:
        print("❌ Environment mismatch detected!")
        print("\n🔧 Fix steps:")
        print("1. Activate the correct environment:")
        print("   conda activate unified_rag")
        print("\n2. Install packages:")
        print("   pip install qdrant-client sentence-transformers streamlit openai neo4j")
        print("\n3. Run streamlit:")
        print("   streamlit run unified_chatbot.py")

def main():
    print("🚀 PYTHON ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    
    check_current_environment()
    check_package_locations()
    check_streamlit_command()
    suggest_fixes()
    
    print(f"\n💡 TIP: If packages show as installed but Streamlit can't find them,")
    print(f"   the issue is likely that Streamlit is running in a different Python environment.")

if __name__ == "__main__":
    main()