#!/usr/bin/env python3
"""
Download Google Gemma-2-2B model for Motion-Agent
This script downloads the Gemma-2-2B-it model from HuggingFace to the correct location.
"""

import os
import sys
import shutil
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import huggingface_hub
        print("✅ huggingface_hub is installed")
        return True
    except ImportError:
        print("❌ huggingface_hub is not installed")
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface-hub")
        try:
            import huggingface_hub
            print("✅ Successfully installed huggingface_hub")
            return True
        except ImportError:
            print("❌ Failed to install huggingface_hub")
            print("Please install manually: pip install huggingface-hub")
            return False

def check_hf_login():
    """Check if user is logged into HuggingFace"""
    from huggingface_hub import HfApi
    
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print("❌ Not logged into HuggingFace")
        print("\nPlease follow these steps:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token (or use existing one)")
        print("3. Run: huggingface-cli login")
        print("4. Enter your token when prompted")
        print("\nAlternatively, set the environment variable:")
        print("export HF_TOKEN=your_token_here")
        return False

def check_gemma_access():
    """Check if user has access to Gemma model"""
    from huggingface_hub import HfApi
    
    api = HfApi()
    model_id = "google/gemma-2-2b-it"
    
    print(f"\n📋 Checking access to {model_id}...")
    print("Note: You need to request access at:")
    print("https://huggingface.co/google/gemma-2-2b-it")
    print("Click 'Request access' and agree to the terms")
    
    # We'll try to download anyway, as the check might fail but download could work
    return True

def download_model():
    """Download the Gemma-2-2B model"""
    from huggingface_hub import snapshot_download
    
    model_id = "google/gemma-2-2b-it"
    target_dir = "scene_motion_planner/gemma2b"
    
    # Check if target directory exists as a file
    if os.path.exists(target_dir) and os.path.isfile(target_dir):
        print(f"🗑️  Removing empty placeholder file: {target_dir}")
        os.remove(target_dir)
    
    # Check if model already exists
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        model_files = os.listdir(target_dir)
        if len(model_files) > 5:  # Rough check for existing model
            print(f"📁 Model directory already exists with {len(model_files)} files")
            response = input("Do you want to re-download? (y/n): ").strip().lower()
            if response != 'y':
                print("✅ Using existing model")
                return True
            else:
                print(f"🗑️  Removing existing model directory...")
                shutil.rmtree(target_dir)
    
    # Download the model
    print(f"\n📥 Downloading {model_id} to {target_dir}")
    print("This may take a while (model size is ~5GB)...")
    print("-" * 50)
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,  # Enable resume for interrupted downloads
            token=os.getenv("HF_TOKEN")  # Use token from environment if available
        )
        print("\n✅ Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nPossible reasons:")
        print("1. You don't have access to the model (request at the HuggingFace page)")
        print("2. Network issues")
        print("3. Insufficient disk space")
        print("4. Invalid HuggingFace token")
        return False

def verify_download():
    """Verify that essential model files are present"""
    target_dir = "scene_motion_planner/gemma2b"
    
    if not os.path.exists(target_dir):
        print("❌ Model directory does not exist")
        return False
    
    required_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json"
    ]
    
    # Check for model weights (could be different formats)
    weight_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "model-00001-of-00002.safetensors"  # For sharded models
    ]
    
    print("\n🔍 Verifying downloaded files...")
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(target_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  ✅ {file} ({size:.2f} MB)")
        else:
            missing_files.append(file)
            print(f"  ❌ {file} (missing)")
    
    # Check for at least one weight file
    weight_found = False
    for weight_file in weight_files:
        file_path = os.path.join(target_dir, weight_file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # GB
            print(f"  ✅ {weight_file} ({size:.2f} GB)")
            weight_found = True
            break
    
    if not weight_found:
        print(f"  ❌ No model weights file found")
        missing_files.append("model weights")
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    # List all files in the directory
    all_files = os.listdir(target_dir)
    print(f"\n📁 Total files downloaded: {len(all_files)}")
    
    return True

def main():
    """Main function to orchestrate the download process"""
    print("=" * 60)
    print("🤖 Gemma-2-2B Model Downloader for Scene Motion Planner")
    print("=" * 60)
    
    # Step 1: Check dependencies
    print("\n[Step 1/5] Checking dependencies...")
    if not check_dependencies():
        return 1
    
    # Step 2: Check HuggingFace login
    print("\n[Step 2/5] Checking HuggingFace login...")
    if not check_hf_login():
        print("\n⚠️  You need to login to HuggingFace first")
        return 1
    
    # Step 3: Check model access
    print("\n[Step 3/5] Checking model access...")
    check_gemma_access()
    
    # Step 4: Download the model
    print("\n[Step 4/5] Downloading model...")
    if not download_model():
        return 1
    
    # Step 5: Verify download
    print("\n[Step 5/5] Verifying download...")
    if not verify_download():
        print("\n⚠️  Download verification failed")
        print("The model may not work correctly")
        return 1
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Gemma-2-2B model is ready to use")
    print("=" * 60)
    print("\nYou can now run the Scene Motion Planner with MotionLLM support:")
    print("  python scene_motion_planner/main.py --motion-gen 'A person walks forward'")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())