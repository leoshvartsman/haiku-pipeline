#!/bin/bash
# Batch convert all haiku book txt files to PDF and EPUB

SERIES_DIR="/Users/leo/Downloads/haiku-generator/haiku_output/series"
FORMATTER="/Users/leo/Downloads/book_formatter/book_formatter.py"
OUTPUT_DIR="/Users/leo/Downloads/book_formatter/output"

# Check if series directory exists
if [ ! -d "$SERIES_DIR" ]; then
    echo "Error: Series directory not found: $SERIES_DIR"
    exit 1
fi

# Count files
count=$(ls -1 "$SERIES_DIR"/*.txt 2>/dev/null | wc -l)
if [ "$count" -eq 0 ]; then
    echo "No .txt files found in $SERIES_DIR"
    exit 1
fi

echo "Found $count txt files to process"
echo "Output will be saved to: $OUTPUT_DIR"
echo "=========================================="

# Process each txt file
for txtfile in "$SERIES_DIR"/*.txt; do
    filename=$(basename "$txtfile")
    echo ""
    echo "Processing: $filename"
    echo "----------------------------------------"
    python3 "$FORMATTER" "$txtfile" --out-dir "$OUTPUT_DIR"
done

echo ""
echo "=========================================="
echo "Batch processing complete!"
echo "Check output in: $OUTPUT_DIR"
