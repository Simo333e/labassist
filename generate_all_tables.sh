#!/bin/bash
# Generate all LaTeX tables for DTULabAssist thesis
# Usage: bash generate_all_tables.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     DTULabAssist - LaTeX Table Generation Script                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
if [ -d ".env/bin" ]; then
    echo "🔧 Activating virtual environment..."
    source .env/bin/activate
else
    echo "⚠️  Virtual environment not found at .env/bin"
    echo "   Using system Python..."
fi

echo ""
echo "📊 Generating YOLO and combined results tables..."
python3 create_yolo_latex_table.py

echo ""
echo "📈 Generating per-class action recognition table..."
python3 create_per_class_latex.py

echo ""
echo "✅ All tables generated successfully!"
echo ""
echo "📁 Output directory: latex_tables/"
echo ""
echo "Generated files:"
ls -1 latex_tables/*.tex | sed 's/^/  ✓ /'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 To use in your LaTeX document:"
echo ""
echo "  \\usepackage{booktabs}"
echo "  \\usepackage{multirow}"
echo "  \\usepackage{graphicx}"
echo ""
echo "  \\input{latex_tables/combined_results.tex}"
echo "  \\input{latex_tables/yolo_results.tex}"
echo "  \\input{latex_tables/per_class_action.tex}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


