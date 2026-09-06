"""
Package entry point so `python -m assistive_navigation` works.
Delegates to main.main(). No pipeline logic here.
"""
from assistive_navigation.main import main

if __name__ == "__main__":
    main()
