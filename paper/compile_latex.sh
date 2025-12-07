#!/bin/bash

# Compilation script for LaTeX documents
# Creates PDF versions of all methodology chapters

echo "========================================="
echo "Compiling LaTeX Methodology Documents"
echo "========================================="

# Directory setup
DOC_DIR="$(dirname "$0")"
OUTPUT_DIR="$DOC_DIR/pdf"
mkdir -p "$OUTPUT_DIR"

# LaTeX files to compile
TEX_FILES=(
    "methodology_complete.tex"
    "chapter01_introduction.tex"
    "chapter03_attention_derivations.tex"
    "chapter05_adversarial_proofs.tex"
)

# Compile each document
for tex_file in "${TEX_FILES[@]}"; do
    if [ -f "$DOC_DIR/$tex_file" ]; then
        echo ""
        echo "Compiling: $tex_file"
        echo "----------------------------------------"
        
        # Extract base name (without .tex extension)
        base_name=$(basename "$tex_file" .tex)
        
        # Change to document directory
        cd "$DOC_DIR"
        
        # First pass
        pdflatex -interaction=nonstopmode -output-directory="$OUTPUT_DIR" "$tex_file" > /dev/null 2>&1
        
        # BibTeX if .bbl file exists
        if [ -f "$OUTPUT_DIR/$base_name.bbl" ] || grep -q "\\bibliography" "$tex_file"; then
            cd "$OUTPUT_DIR"
            bibtex "$base_name" > /dev/null 2>&1
            cd "$DOC_DIR"
        fi
        
        # Second pass for references
        pdflatex -interaction=nonstopmode -output-directory="$OUTPUT_DIR" "$tex_file" > /dev/null 2>&1
        
        # Third pass to ensure all references resolved
        pdflatex -interaction=nonstopmode -output-directory="$OUTPUT_DIR" "$tex_file" > /dev/null 2>&1
        
        # Check if PDF was created
        if [ -f "$OUTPUT_DIR/$base_name.pdf" ]; then
            echo "✓ Successfully compiled: $base_name.pdf"
        else
            echo "✗ Failed to compile: $tex_file"
        fi
        
        # Clean up auxiliary files
        rm -f "$OUTPUT_DIR/$base_name.aux"
        rm -f "$OUTPUT_DIR/$base_name.log"
        rm -f "$OUTPUT_DIR/$base_name.out"
        rm -f "$OUTPUT_DIR/$base_name.toc"
        rm -f "$OUTPUT_DIR/$base_name.bbl"
        rm -f "$OUTPUT_DIR/$base_name.blg"
    else
        echo "✗ File not found: $tex_file"
    fi
done

echo ""
echo "========================================="
echo "Compilation complete!"
echo "PDF files saved to: $OUTPUT_DIR"
echo "========================================="

# List generated PDFs
echo ""
echo "Generated PDF files:"
ls -lh "$OUTPUT_DIR"/*.pdf 2>/dev/null || echo "No PDF files generated."

