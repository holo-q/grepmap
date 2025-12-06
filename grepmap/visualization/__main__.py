"""Entry point for python -m grepmap.visualization"""
from .cli import main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="grepmap graph visualizer")
    parser.add_argument("root", nargs="?", default=".", help="Repository root")
    parser.add_argument("-p", "--port", type=int, default=8765, help="HTTP port")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    main(
        root=args.root,
        port=args.port,
        no_open=args.no_open,
        verbose=args.verbose
    )
