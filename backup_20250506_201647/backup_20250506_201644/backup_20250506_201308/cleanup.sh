#!/bin/bash

echo "Organizing project files..."

# Create directories if they don't exist
mkdir -p adapters
mkdir -p scripts
mkdir -p docs

# Move adapter files to adapters directory
echo "Moving adapter files to adapters/ directory..."
for file in adapter.py adapter_step1.py adapter_step2.py adapter_step3.py adapter_step1_fixed.py \
            adapter_fix.py adapter_working.py stream_fixed_adapter.py flexible_adapter.py \
            complete_adapter.py final_adapter.py; do
    if [ -f "$file" ]; then
        cp "$file" "adapters/$file"
        echo "  Copied $file to adapters/"
    fi
done

# Keep enhanced_adapter.py in the root as the primary adapter
cp enhanced_adapter.py adapters/enhanced_adapter.py
echo "Kept enhanced_adapter.py as the primary adapter in root"

# Move script files to scripts directory
echo "Moving script files to scripts/ directory..."
for file in run_adapter.sh run_flexible.sh run_stream_fixed.sh run_complete.sh run_final.sh \
            full_integration.sh fix_frontend.js patch_index.html; do
    if [ -f "$file" ]; then
        cp "$file" "scripts/$file"
        echo "  Copied $file to scripts/"
    fi
done

# Keep run_enhanced.sh and simple_integration.sh in the root for easy access
cp run_enhanced.sh scripts/run_enhanced.sh
cp simple_integration.sh scripts/simple_integration.sh
echo "Kept run_enhanced.sh and simple_integration.sh in root for easy access"

# Move documentation to docs directory
echo "Moving documentation to docs/ directory..."
cp README_INTEGRATION.md docs/
cp INTEGRATION.md docs/ 2>/dev/null || echo "  INTEGRATION.md not found"

# Create symlinks for essential files to maintain compatibility
echo "Creating symlinks for essential files..."
ln -sf enhanced_adapter.py adapter.py
ln -sf run_enhanced.sh run.sh

echo "Creating cleanup documentation..."
cat > README.md << 'EOF'
# AI Risk Repository Chatbot

A system for interacting with the MIT AI Risk Repository using modern RAG techniques
and a React frontend.

## Quick Start

```bash
# Set up the frontend
./simple_integration.sh

# Run the enhanced adapter
./run_enhanced.sh
```

Then access the chatbot at http://localhost:5000

## Project Structure

- `enhanced_adapter.py` - The primary adapter with all features
- `run_enhanced.sh` - Script to run the enhanced adapter
- `simple_integration.sh` - Script to set up the frontend

### Core Components

- `vector_store.py` - Retrieval-Augmented Generation system
- `monitor.py` - Query classification and analysis
- `gemini_model.py` - LLM integration for response generation

### Directories

- `adapters/` - Alternative adapter implementations
- `scripts/` - Helper scripts and utilities
- `docs/` - Documentation files
- `info_files/` - Repository documents
- `github-frontend-build/` - Built frontend files

## Documentation

See [README_INTEGRATION.md](docs/README_INTEGRATION.md) for detailed integration instructions.

## Features

- Robust RAG implementation with hybrid search
- Clickable document citations and source viewer
- Domain-specific query handling
- Multi-level fallback mechanisms
EOF

echo "Project reorganization complete!"
echo "Use enhanced_adapter.py as the primary adapter and run_enhanced.sh to start the server."