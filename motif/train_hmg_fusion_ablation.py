"""V3 controlled entry point. Historical versions remain in snapshots/source.

Defaults to smoke validation; formal stages require explicit unresolved settings.
"""
from .train_hmg_design import main

if __name__ == '__main__':
    main()
