"""Command-line entry point. Partial batch failures do not discard successful items."""

import argparse
import json
import sys
from pathlib import Path

from .core import (
    Product, ValidationError, InsufficientDataError, load_records, markdown,
)
from .llm import APIError, LLMClient, generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Черновики товарных карточек через LLM API")
    parser.add_argument("input", type=Path, help="JSON: один товар или массив товаров")
    parser.add_argument("--output", type=Path, help="Выходной файл, по умолчанию stdout")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() == args.input.resolve():
            raise ValidationError("Выходной файл не должен совпадать с входным")
        if args.input.stat().st_size > 1_000_000:
            raise ValidationError("Входной файл превышает 1 MB")
        products = load_records(args.input.read_text(encoding="utf-8-sig"))
        client = LLMClient()
        results = []
        failed = False
        for index, raw in enumerate(products):
            product = None
            try:
                product = Product.parse(raw)
                card = generate(product, client)
                results.append({
                    "index": index,
                    "mode": "llm_api",
                    "model": client.model,
                    "card": card,
                })
            except (ValidationError, APIError) as exc:
                failed = True
                results.append({
                    "index": index, "product_name": product.name if product else None,
                    "status": "needs_clarification" if isinstance(exc, InsufficientDataError) else "error",
                    "error": str(exc), "missing_fields": product.missing_fields() if product else [],
                })
                if isinstance(exc, InsufficientDataError):
                    results[-1]["missing_fields"].append(
                        "description/specs: нужны дополнительные подтвержденные сведения о назначении и функциях"
                    )
        output = (
            markdown(results) if args.format == "markdown"
            else json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        return 1 if failed else 0
    except (ValidationError, APIError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
