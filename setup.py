#!/usr/bin/env python3
"""
Setup script to download the UD English-EWT dataset.
"""

import os
import urllib.request
import sys


def download_file(url, filepath):
    """Download a file from URL."""
    try:
        print(f"Downloading {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ Downloaded {filepath}")
        return True
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        return False


def main():
    # Create data directory
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Dataset URLs
    base_url = 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/'
    files = ['en_ewt-ud-train.conllu', 'en_ewt-ud-test.conllu']
    
    print("=" * 60)
    print("Downloading UD English-EWT Dataset")
    print("=" * 60 + "\n")
    
    success_count = 0
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"✓ {filename} already exists")
            success_count += 1
        else:
            url = base_url + filename
            if download_file(url, filepath):
                success_count += 1
    
    print("\n" + "=" * 60)
    if success_count == len(files):
        print("✓ All files downloaded successfully!")
        print("\nNext steps:")
        print("  cd src")
        print("  python main.py --train ../data/en_ewt-ud-train.conllu --test ../data/en_ewt-ud-test.conllu")
    else:
        print("⚠ Some files failed to download")
        print("\nManual download:")
        print("  Visit: https://github.com/UniversalDependencies/UD_English-EWT")
        print("  Download en_ewt-ud-train.conllu and en_ewt-ud-test.conllu")
        print("  Place them in the data/ directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
