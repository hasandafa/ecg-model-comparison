"""
Diagnostic script to check config.yaml file
Run this to diagnose configuration issues
"""

import os
import yaml
from pathlib import Path

def check_config():
    """Check if config.yaml exists and is valid."""
    
    print("\n" + "="*70)
    print("Configuration File Diagnostic")
    print("="*70 + "\n")
    
    # Check if file exists
    config_path = Path('config.yaml')
    
    print(f"1. Checking if config.yaml exists...")
    if not config_path.exists():
        print(f"   ❌ File not found: {config_path.absolute()}")
        print(f"\n💡 Solution:")
        print(f"   The config.yaml file is missing. You need to create it.")
        print(f"   Copy the config.yaml content from the artifacts provided.")
        return False
    else:
        print(f"   ✅ File exists: {config_path.absolute()}")
    
    # Check file size
    print(f"\n2. Checking file size...")
    file_size = config_path.stat().st_size
    print(f"   File size: {file_size} bytes")
    
    if file_size == 0:
        print(f"   ❌ File is empty!")
        print(f"\n💡 Solution:")
        print(f"   Your config.yaml file is empty.")
        print(f"   Copy the complete config.yaml content from the artifacts.")
        return False
    elif file_size < 100:
        print(f"   ⚠️  File is very small ({file_size} bytes)")
        print(f"   Expected size: ~15-20 KB")
    else:
        print(f"   ✅ File has content")
    
    # Try to read the file
    print(f"\n3. Checking file content...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"   First 200 characters:")
        print(f"   {'-'*66}")
        print(f"   {content[:200]}")
        print(f"   {'-'*66}")
        
        if not content.strip():
            print(f"   ❌ File content is empty or whitespace only")
            return False
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False
    
    # Try to parse YAML
    print(f"\n4. Parsing YAML...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print(f"   ❌ YAML parsed but returned None")
            print(f"   This usually means the file has only comments or is improperly formatted")
            print(f"\n💡 Solution:")
            print(f"   Make sure your config.yaml has actual key-value pairs, not just comments.")
            print(f"   The file should start with configuration keys like 'seed:', 'data:', etc.")
            return False
        
        print(f"   ✅ YAML parsed successfully")
        print(f"   Type: {type(config)}")
        
    except yaml.YAMLError as e:
        print(f"   ❌ YAML parsing error: {e}")
        print(f"\n💡 Solution:")
        print(f"   Your config.yaml has invalid YAML syntax.")
        print(f"   Common issues:")
        print(f"   - Mixed tabs and spaces (use spaces only)")
        print(f"   - Incorrect indentation")
        print(f"   - Missing colons after keys")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False
    
    # Check required keys
    print(f"\n5. Checking required keys...")
    required_keys = ['seed', 'data', 'split', 'preprocessing', 'training', 'models']
    missing_keys = []
    
    for key in required_keys:
        if key in config:
            print(f"   ✅ '{key}' found")
        else:
            print(f"   ❌ '{key}' missing")
            missing_keys.append(key)
    
    if missing_keys:
        print(f"\n   Missing keys: {missing_keys}")
        print(f"\n💡 Solution:")
        print(f"   Your config.yaml is incomplete.")
        print(f"   Make sure you copied the ENTIRE config.yaml content from the artifacts.")
        return False
    
    # Show summary
    print(f"\n6. Configuration Summary:")
    print(f"   Seed: {config.get('seed', 'NOT SET')}")
    print(f"   Models: {config.get('models', {}).get('enabled_models', 'NOT SET')}")
    print(f"   CV Folds: {config.get('split', {}).get('n_folds', 'NOT SET')}")
    
    print(f"\n{'='*70}")
    print(f"✅ Configuration file is valid!")
    print(f"{'='*70}\n")
    
    return True


if __name__ == '__main__':
    success = check_config()
    
    if not success:
        print("\n" + "="*70)
        print("FAILED - Please fix config.yaml")
        print("="*70)
        print("\nQuick Fix Steps:")
        print("1. Delete your current config.yaml (if it exists)")
        print("2. Create a new config.yaml file")
        print("3. Copy the COMPLETE content from the 'config.yaml' artifact")
        print("4. Make sure you copy EVERYTHING (should be ~500+ lines)")
        print("5. Save the file")
        print("6. Run this diagnostic script again: python check_config.py")
        print("="*70 + "\n")
    else:
        print("You can now run: python run_pipeline.py --step patient_mapping")