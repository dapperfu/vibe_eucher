#!/bin/bash
# Generate Model Input Documentation Script
# 
# This script generates human-readable documentation of all 236 model input features.
# 
# Setup:
#   1. Ensure virtual environment is activated: source venv_vibe_eucher/bin/activate
#   2. Ensure dependencies are installed: pip install -e .
#   3. Run this script: ./scripts/generate_docs.sh
#
# What it does:
#   - Generates markdown documentation: docs/model_inputs.md
#   - Generates interactive HTML report: docs/model_inputs.html
#   - Documents all 236 input features with descriptions
#   - Includes decision weight system explanation
#
# Output:
#   - docs/model_inputs.md (markdown reference)
#   - docs/model_inputs.html (interactive HTML report)

set -e  # Exit on error

FORMAT=${1:-"both"}  # Default "both", can override: ./generate_docs.sh markdown

echo "=========================================="
echo "Generate Model Input Documentation"
echo "=========================================="
echo ""
echo "Setup:"
echo "  - Virtual environment: venv_vibe_eucher"
echo "  - Output format: ${FORMAT}"
echo "  - Output directory: docs/"
echo ""
echo "Features documented:"
echo "  - Hand encoding: 120 features (5 cards × 24 one-hot)"
echo "  - Turned card: 24 features"
echo "  - Trump suit: 4 features"
echo "  - Led suit: 4 features"
echo "  - Trick cards: 72 features (3 cards × 24 one-hot)"
echo "  - Positional: 12 features"
echo "  - Total: 236 features"
echo ""
echo "Command to run:"
echo "  python scripts/generate_input_docs.py --format ${FORMAT}"
echo ""
echo "Generating documentation..."
echo ""

# Run the documentation generation
python scripts/generate_input_docs.py --format "${FORMAT}"

echo ""
echo "=========================================="
echo "Documentation generation complete!"
echo "=========================================="
echo ""
if [ "${FORMAT}" = "both" ] || [ "${FORMAT}" = "markdown" ]; then
    echo "  - Markdown: docs/model_inputs.md"
fi
if [ "${FORMAT}" = "both" ] || [ "${FORMAT}" = "html" ]; then
    echo "  - HTML: docs/model_inputs.html"
fi
echo ""

