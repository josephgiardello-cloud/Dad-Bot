#!/usr/bin/env python3
"""
Quick launcher — double-click friendly.
Run: python launch.py

Routes through app_runtime.main() so all startup safety checks,
contract validation, and resource guards are always enforced.
"""

import sys


def main():
    print("🧔 Starting DadBot...")
    try:
        from dadbot.app_runtime import main as app_main
        raise SystemExit(app_main())
    except KeyboardInterrupt:
        print("\n👋 DadBot stopped.")
    except SystemExit:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
