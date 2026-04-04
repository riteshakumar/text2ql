from __future__ import annotations

import argparse
import json

from text2ql.core import Text2QL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text to Query Language CLI")
    parser.add_argument("text", help="Natural language request")
    parser.add_argument("--target", default="graphql", help="Target query language")
    parser.add_argument(
        "--schema",
        default="",
        help="Schema as JSON string, e.g. '{\"entities\":[\"users\"],\"fields\":[\"id\",\"name\"]}'",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    schema = json.loads(args.schema) if args.schema else None
    service = Text2QL()
    result = service.generate(text=args.text, target=args.target, schema=schema)

    print(result.query)


if __name__ == "__main__":
    main()
