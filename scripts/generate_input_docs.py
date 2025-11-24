"""Script to generate model input documentation."""

import argparse
from pathlib import Path

from src.training.input_documentation import InputDocumentationGenerator


def main() -> None:
    """Generate model input documentation."""
    parser = argparse.ArgumentParser(description="Generate model input documentation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs",
        help="Output directory for documentation (default: docs/)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["both", "markdown", "html"],
        default="both",
        help="Output format: markdown, html, or both (default: both)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = InputDocumentationGenerator()

    print("Generating model input documentation...")

    if args.format in ["both", "markdown"]:
        markdown_path = output_dir / "model_inputs.md"
        generator.generate_markdown_docs(markdown_path)
        print(f"Markdown documentation saved to: {markdown_path}")

    if args.format in ["both", "html"]:
        html_path = output_dir / "model_inputs.html"
        generator.generate_html_docs(html_path)
        print(f"HTML documentation saved to: {html_path}")

    print("Documentation generation complete!")


if __name__ == "__main__":
    main()

