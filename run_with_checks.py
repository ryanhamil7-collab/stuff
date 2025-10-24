#!/usr/bin/env python3
"""
Launcher wrapper that runs system checks before starting the trading system.
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.system_checks import run_system_checks


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Launch Autonomous Trading System with pre-flight checks'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip system checks and launch directly'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors in system checks'
    )
    parser.add_argument(
        '--checks-only',
        action='store_true',
        help='Run checks only, do not launch system'
    )
    
    args, launcher_args = parser.parse_known_args()
    
    if not args.skip_checks:
        print("Running pre-flight system checks...\n")
        checks_passed = run_system_checks(
            strict_mode=args.strict,
            skip_optional=False
        )
        
        if not checks_passed:
            print("\n❌ System checks failed. Please fix errors before launching.")
            print("To skip checks (not recommended), use --skip-checks flag.")
            sys.exit(1)
        
        print("\n✅ All system checks passed!")
    
    if args.checks_only:
        print("\nChecks-only mode: exiting without launching system.")
        sys.exit(0)
    
    print("\nLaunching Autonomous Trading System...\n")
    print("=" * 70)
    
    try:
        import subprocess
        launcher_cmd = [sys.executable, 'launcher.py'] + launcher_args
        result = subprocess.run(launcher_cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error launching system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
