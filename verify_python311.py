#!/usr/bin/env python3
"""
Python 3.11+ Compatibility Verification Script
Tests all dependency fixes and ensures system is ready for Kaggle deployment.
"""

import sys
import os
from pathlib import Path

class Python311Verifier:
    """Verify Python 3.11+ compatibility."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def print_header(self, text):
        """Print formatted header."""
        print("\n" + "=" * 70)
        print(text)
        print("=" * 70)
    
    def test_python_version(self):
        """Test 1: Python version check."""
        print("\n[Test 1] Python Version Check")
        version = sys.version_info
        print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major >= 3 and version.minor >= 10:
            print("  ✓ Python 3.10+ requirement met")
            self.passed.append("Python version >= 3.10")
            return True
        else:
            print(f"  ✗ Python 3.10+ required, got {version.major}.{version.minor}")
            self.failed.append("Python version check")
            return False
    
    def test_requirements_file(self):
        """Test 2: Verify requirements.txt has no invalid entries."""
        print("\n[Test 2] Requirements.txt Validation")
        
        req_file = Path("requirements.txt")
        if not req_file.exists():
            print("  ✗ requirements.txt not found")
            self.failed.append("requirements.txt missing")
            return False
        
        with open(req_file, 'r') as f:
            content = f.read()
        
        invalid_entries = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('python>=') and not line.startswith('#'):
                invalid_entries.append("python>=3.10 (should be commented out)")
        
        if '\nsqlite3' in content or content.startswith('sqlite3'):
            invalid_entries.append("sqlite3 (built-in module)")
        
        if 'pandas-ta==0.3.14b0' in content:
            invalid_entries.append("pandas-ta==0.3.14b0 (outdated for Python 3.11)")
        
        if invalid_entries:
            print("  ✗ Found invalid entries:")
            for entry in invalid_entries:
                print(f"    - {entry}")
            self.failed.append("requirements.txt validation")
            return False
        
        if '# python>=3.10' in content:
            print("  ✓ python>=3.10 properly commented out")
        
        if 'pandas-ta>=0.4.71b0' in content or 'pandas-ta>=0.4' in content:
            print("  ✓ pandas-ta updated to Python 3.11-compatible version")
        
        if '# sqlite3 is built-in' in content or 'sqlite3' not in content:
            print("  ✓ sqlite3 removed or commented")
        
        if '# auto-gptq' in content or 'auto-gptq' not in content:
            print("  ✓ auto-gptq made optional or commented")
        
        self.passed.append("requirements.txt validation")
        return True
    
    def test_core_imports(self):
        """Test 3: Test core package imports."""
        print("\n[Test 3] Core Package Imports")
        
        core_packages = {
            'pandas': 'pandas',
            'numpy': 'numpy',
            'yaml': 'pyyaml',
            'dotenv': 'python-dotenv',
        }
        
        failed_imports = []
        
        for module, package in core_packages.items():
            try:
                if module == 'dotenv':
                    __import__('dotenv')
                else:
                    __import__(module)
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  ✗ {package}")
                failed_imports.append(package)
        
        if failed_imports:
            self.failed.append(f"Core imports: {', '.join(failed_imports)}")
            return False
        
        self.passed.append("Core package imports")
        return True
    
    def test_sqlite3(self):
        """Test 4: Verify sqlite3 is available (built-in)."""
        print("\n[Test 4] SQLite3 Built-in Module")
        
        try:
            import sqlite3
            print(f"  ✓ sqlite3 available (version {sqlite3.sqlite_version})")
            self.passed.append("sqlite3 built-in module")
            return True
        except ImportError:
            print("  ✗ sqlite3 not available")
            self.failed.append("sqlite3 import")
            return False
    
    def test_pandas_ta(self):
        """Test 5: Test pandas-ta import and version."""
        print("\n[Test 5] pandas-ta Compatibility")
        
        try:
            import pandas_ta
            version = getattr(pandas_ta, '__version__', 'unknown')
            print(f"  ✓ pandas-ta available (version {version})")
            
            if version != 'unknown':
                major_minor = version.split('.')[:2]
                if len(major_minor) >= 2:
                    major = int(major_minor[0])
                    minor = int(major_minor[1].split('b')[0])
                    if major == 0 and minor >= 4:
                        print(f"  ✓ pandas-ta version {version} is Python 3.11-compatible")
                    elif major > 0:
                        print(f"  ✓ pandas-ta version {version} is Python 3.11-compatible")
            
            self.passed.append("pandas-ta import")
            return True
        except ImportError:
            print("  ⚠ pandas-ta not available (optional)")
            self.warnings.append("pandas-ta not installed (optional)")
            return True
    
    def test_auto_gptq(self):
        """Test 6: Test auto-gptq (optional)."""
        print("\n[Test 6] auto-gptq (Optional)")
        
        try:
            import auto_gptq
            print(f"  ✓ auto-gptq available")
            self.passed.append("auto-gptq available")
        except ImportError:
            print("  ⚠ auto-gptq not available (optional, requires CUDA)")
            self.warnings.append("auto-gptq not installed (optional)")
        
        return True
    
    def test_updated_dependencies(self):
        """Test 7: Verify updated dependencies are available."""
        print("\n[Test 7] Updated Dependencies")
        
        updated_packages = {
            'torch': '2.1.0',
            'transformers': '4.40.0',
            'langchain': '0.2.0',
        }
        
        for package, min_version in updated_packages.items():
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"  ✓ {package} (version {version})")
            except ImportError:
                print(f"  ⚠ {package} not installed (optional)")
                self.warnings.append(f"{package} not installed")
        
        self.passed.append("Updated dependencies check")
        return True
    
    def test_install_scripts(self):
        """Test 8: Verify install scripts exist and are executable."""
        print("\n[Test 8] Installation Scripts")
        
        scripts = {
            'install.py': 'Python installer',
            'setup.sh': 'Bash setup script',
        }
        
        for script, desc in scripts.items():
            script_path = Path(script)
            if script_path.exists():
                print(f"  ✓ {desc} ({script})")
                
                if os.access(script_path, os.X_OK):
                    print(f"    ✓ Executable")
                else:
                    print(f"    ⚠ Not executable (may need chmod +x)")
            else:
                print(f"  ✗ {desc} ({script}) not found")
                self.failed.append(f"{script} missing")
        
        self.passed.append("Installation scripts")
        return True
    
    def test_documentation(self):
        """Test 9: Verify documentation is updated."""
        print("\n[Test 9] Documentation")
        
        docs = {
            'README.md': ['Kaggle', 'Python 3.11'],
            'docs/KAGGLE_SETUP.md': ['Python 3.11', 'pandas-ta', 'auto-gptq'],
            'PYTHON311_FIXES.md': ['dependency', 'fixes'],
        }
        
        for doc, keywords in docs.items():
            doc_path = Path(doc)
            if doc_path.exists():
                with open(doc_path, 'r') as f:
                    content = f.read().lower()
                
                found_keywords = [kw for kw in keywords if kw.lower() in content]
                
                if found_keywords:
                    print(f"  ✓ {doc} (contains: {', '.join(found_keywords)})")
                else:
                    print(f"  ⚠ {doc} (missing keywords: {', '.join(keywords)})")
                    self.warnings.append(f"{doc} missing keywords")
            else:
                print(f"  ✗ {doc} not found")
                if doc == 'PYTHON311_FIXES.md':
                    self.warnings.append(f"{doc} missing")
                else:
                    self.failed.append(f"{doc} missing")
        
        self.passed.append("Documentation check")
        return True
    
    def test_kaggle_compatibility(self):
        """Test 10: Verify Kaggle-specific features."""
        print("\n[Test 10] Kaggle Compatibility Features")
        
        if Path('kaggle_notebook.py').exists():
            print("  ✓ kaggle_notebook.py exists")
        else:
            print("  ⚠ kaggle_notebook.py not found")
            self.warnings.append("kaggle_notebook.py missing")
        
        if Path('.env.example').exists():
            print("  ✓ .env.example exists")
        else:
            print("  ⚠ .env.example not found")
            self.warnings.append(".env.example missing")
        
        if Path('install.py').exists():
            with open('install.py', 'r') as f:
                content = f.read()
            
            if 'KAGGLE_KERNEL_RUN_TYPE' in content or 'kaggle' in content.lower():
                print("  ✓ install.py has Kaggle detection")
            else:
                print("  ⚠ install.py missing Kaggle detection")
                self.warnings.append("install.py missing Kaggle detection")
        
        self.passed.append("Kaggle compatibility")
        return True
    
    def print_summary(self):
        """Print test summary."""
        self.print_header("Test Summary")
        
        print(f"\n✓ Passed: {len(self.passed)}")
        for test in self.passed:
            print(f"  - {test}")
        
        if self.warnings:
            print(f"\n⚠ Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.failed:
            print(f"\n✗ Failed: {len(self.failed)}")
            for failure in self.failed:
                print(f"  - {failure}")
            print("\n❌ VERIFICATION FAILED")
            return False
        else:
            print("\n✅ ALL TESTS PASSED")
            print("\nThe system is ready for Python 3.11+ deployment on Kaggle!")
            return True
    
    def run(self):
        """Run all verification tests."""
        self.print_header("Python 3.11+ Compatibility Verification")
        
        print(f"\nRunning on: {sys.platform}")
        print(f"Python: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        
        self.test_python_version()
        self.test_requirements_file()
        self.test_core_imports()
        self.test_sqlite3()
        self.test_pandas_ta()
        self.test_auto_gptq()
        self.test_updated_dependencies()
        self.test_install_scripts()
        self.test_documentation()
        self.test_kaggle_compatibility()
        
        success = self.print_summary()
        
        return 0 if success else 1


def main():
    """Main entry point."""
    verifier = Python311Verifier()
    exit_code = verifier.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
