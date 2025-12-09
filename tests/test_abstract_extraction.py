# test_abstract_extraction.py
"""
Test script for abstract_extraction module
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def test_excel_reading():
    """Test reading Excel file and column detection"""
    print("=" * 60)
    print("Testing Abstract Extraction - Excel Reading")
    print("=" * 60)
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError:
        print("❌ pandas not available. Install: pip install pandas openpyxl")
        return False
    
    # Test file path
    excel_path = PROJECT_ROOT / "data/papers/superalloy_research.xlsx"
    
    if not excel_path.exists():
        print(f"❌ Excel file not found: {excel_path}")
        return False
    
    print(f"✅ Found Excel file: {excel_path}")
    
    try:
        # Read Excel
        df = pd.read_excel(excel_path)
        print(f"✅ Successfully read Excel file")
        print(f"   - Total rows: {len(df)}")
        print(f"   - Total columns: {len(df.columns)}")
        
        # Show columns
        print("\nColumns found:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        # Test column mapping
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'title' in col_lower:
                column_mapping['title'] = col
            elif 'abstract' in col_lower:
                column_mapping['abstract'] = col
            elif 'journal' in col_lower:
                column_mapping['journal'] = col
            elif 'year' in col_lower:
                column_mapping['year'] = col
            elif 'author' in col_lower:
                column_mapping['author'] = col
            elif 'doi' in col_lower:
                column_mapping['doi'] = col
        
        print("\nColumn mapping detected:")
        for key, val in column_mapping.items():
            print(f"   {key}: '{val}'")
        
        # Check required column
        if 'abstract' not in column_mapping:
            print("\n❌ ERROR: No 'Abstract' column found!")
            return False
        
        print("\n✅ Required 'Abstract' column found")
        
        # Sample data
        abstract_col = column_mapping['abstract']
        non_empty_abstracts = df[abstract_col].notna().sum()
        print(f"\nAbstract statistics:")
        print(f"   - Non-empty abstracts: {non_empty_abstracts}")
        print(f"   - Empty abstracts: {len(df) - non_empty_abstracts}")
        
        # Show first abstract sample
        first_abstract = None
        for idx, row in df.iterrows():
            if pd.notna(row[abstract_col]):
                first_abstract = str(row[abstract_col])
                print(f"\nSample abstract (first 200 chars):")
                print(f"   {first_abstract[:200]}...")
                break
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error reading Excel: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\n" + "=" * 60)
    print("Testing Configuration Loading")
    print("=" * 60)
    
    try:
        import yaml
        print("✅ yaml module imported")
        
        settings_path = PROJECT_ROOT / "config/settings.yaml"
        if not settings_path.exists():
            print(f"❌ Settings file not found: {settings_path}")
            return False
        
        print(f"✅ Found settings file: {settings_path}")
        
        with open(settings_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ Settings loaded successfully")
        
        # Check abstract_extraction config
        if 'abstract_extraction' in config:
            print("\n✅ abstract_extraction configuration found:")
            for key, val in config['abstract_extraction'].items():
                print(f"   {key}: {val}")
        else:
            print("\n⚠️  abstract_extraction not in config (will use defaults)")
        
        # Check prompt file
        prompt_path = PROJECT_ROOT / "config/prompts/text_to_graph.md"
        if prompt_path.exists():
            print(f"\n✅ Prompt file found: {prompt_path}")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
            print(f"   Prompt length: {len(prompt_content)} characters")
        else:
            print(f"\n❌ Prompt file not found: {prompt_path}")
            return False
        
        print("\n" + "=" * 60)
        print("✅ CONFIGURATION TESTS PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_import():
    """Test importing the module"""
    print("\n" + "=" * 60)
    print("Testing Module Import")
    print("=" * 60)
    
    try:
        from core.pipeline_qwen import abstract_extraction
        print("✅ abstract_extraction module imported successfully")
        
        # Check functions exist
        required_functions = [
            'setup_logging',
            'load_config_and_prompt',
            'extract_abstracts_from_excel',
            'prepare_batch_requests',
            'run_abstract_extraction'
        ]
        
        for func_name in required_functions:
            if hasattr(abstract_extraction, func_name):
                print(f"   ✅ {func_name} found")
            else:
                print(f"   ❌ {func_name} not found")
                return False
        
        print("\n" + "=" * 60)
        print("✅ MODULE IMPORT TESTS PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    print("🧪 Abstract Extraction Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Excel Reading", test_excel_reading()))
    results.append(("Configuration", test_config_loading()))
    results.append(("Module Import", test_import()))
    
    # Summary
    print("\n\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Ready to run abstract_extraction.py")
    else:
        print("⚠️  SOME TESTS FAILED - Please fix issues before running")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)

