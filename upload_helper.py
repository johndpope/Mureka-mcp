#!/usr/bin/env python3
"""
Mureka MCP Upload Helper

Simple command-line tool to upload files to Mureka using MCP.
Usage: python upload_helper.py <file_path> [--purpose <purpose>]

Note: This script is designed to be called from Claude Code with MCP tools available.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    """
    This script is meant to be called from Claude Code with MCP tools available.
    It just validates the file and provides instructions.
    """
    parser = argparse.ArgumentParser(description="Upload files to Mureka via MCP")
    parser.add_argument("file_path", help="Path to file to upload")
    parser.add_argument("--purpose", default="fine-tuning", 
                       help="Upload purpose (default: fine-tuning)")
    parser.add_argument("--name", help="Custom name for upload")
    
    args = parser.parse_args()
    
    # Convert to absolute path
    file_path = os.path.abspath(args.file_path)
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    
    file_size = os.path.getsize(file_path)
    file_name = Path(file_path).name
    upload_name = args.name or file_name
    
    print(f"File to upload: {file_name}")
    print(f"Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"Upload name: {upload_name}")
    print(f"Purpose: {args.purpose}")
    print()
    print("To upload this file, use Claude Code with the Mureka MCP tools:")
    
    # Check if this is a reference purpose
    reference_purposes = ['reference', 'vocal', 'melody', 'instrumental', 'voice']
    if args.purpose in reference_purposes:
        print(f"For reference purposes, use: mcp__mureka-mcp__upload_reference_file")
        print(f"Parameters:")
        print(f"  file_path: {file_path}")
        print(f"  purpose: {args.purpose}")
        print()
        print("Purpose requirements:")
        print("  - reference: mp3/m4a, exactly 30 seconds (excess trimmed)")
        print("  - vocal: mp3/m4a, 15-30 seconds (excess trimmed)")
        print("  - melody: mp3/m4a/mid, 5-60 seconds (excess trimmed)")
        print("  - instrumental: mp3/m4a, exactly 30 seconds (excess trimmed)")
        print("  - voice: mp3/m4a, 5-15 seconds (excess trimmed)")
    elif file_size <= 10 * 1024 * 1024:  # 10MB
        print(f"For fine-tuning, use: mcp__mureka-mcp__upload_file")
        print(f"Parameters:")
        print(f"  file_path: {file_path}")
        print(f"  purpose: {args.purpose}")
        if args.name:
            print(f"  upload_name: {upload_name}")
    else:
        print("File is larger than 10MB, use multi-part upload for fine-tuning:")
        print("1. mcp__mureka-mcp__create_upload (don't specify bytes parameter)")
        print("2. mcp__mureka-mcp__add_upload_part (multiple times for chunks)")
        print("3. mcp__mureka-mcp__complete_upload")
    
    print()
    print("Note: Make sure MUREKA_API_KEY environment variable is set.")




if __name__ == "__main__":
    main()