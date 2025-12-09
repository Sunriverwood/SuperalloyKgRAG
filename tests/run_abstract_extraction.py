# run_abstract_extraction_example.py
"""
Example script to run abstract extraction
"""
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import and run
from core.pipeline_qwen.abstract_extraction import run_abstract_extraction

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Abstract Extraction Pipeline")
    print("=" * 60)
    print("\nThis will:")
    print("1. Read abstracts from Excel file")
    print("2. Generate abstract_units.jsonl")
    print("3. Create batch extraction requests")
    print("4. Submit to Qwen API")
    print("5. Process results into extracted_abstract_graph.jsonl")
    print("\n" + "=" * 60)
    
    # Make sure API key is set
    import os
    if not os.getenv("QWEN_API_KEY"):
        print("\n⚠️  WARNING: QWEN_API_KEY not set in environment")
        print("Set it with: $env:QWEN_API_KEY='your-key'")
        print("\nContinuing anyway (will fail at API call)...")
    
    print("\n")
    
    # Run the extraction
    run_abstract_extraction()
    
    print("\n" + "=" * 60)
    print("Pipeline completed. Check logs for details.")
    print("=" * 60)

