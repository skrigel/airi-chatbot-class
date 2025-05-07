# AI Risk Repository Chatbot Integration Guide

This document explains how to integrate the enhanced backend with the GitHub frontend UI.

## Overview

This project integrates:

1. **Enhanced Backend**
   - Robust RAG implementation with hybrid search
   - Multi-level fallback mechanisms
   - Domain-specific query handling
   - Vector store, query monitoring, and LLM components

2. **GitHub Frontend**
   - Modern React interface with TypeScript and Tailwind CSS
   - Streaming response support
   - Conversation management

## File Structure

The integration uses several adapter implementations with increasing functionality:

### Core Components

- `vector_store.py` - Retrieval-Augmented Generation system
- `monitor.py` - Query classification and analysis
- `gemini_model.py` - LLM integration for response generation

### Adapter Options

| Adapter | Description | Script |
|---------|-------------|--------|
| **enhanced_adapter.py** | **RECOMMENDED**: Full integration with clickable citations | `run_enhanced.sh` |
| final_adapter.py | Complete integration with robust compatibility | `run_final.sh` |
| complete_adapter.py | Full backend integration | `run_complete.sh` |
| stream_fixed_adapter.py | Fixed streaming support | `run_stream_fixed.sh` |
| flexible_adapter.py | Port-flexible implementation | `run_flexible.sh` |
| adapter_step1_fixed.py | Basic adapter with simple RAG | `run_adapter.sh step1` |

### Integration Scripts

- `simple_integration.sh` - One-step integration setup
- `run_enhanced.sh` - Run the enhanced adapter (recommended)

## Quick Start

For the fastest, most complete integration:

```bash
# Step 1: Run the integration script to set up the frontend
./simple_integration.sh

# Step 2: Run the enhanced adapter
./run_enhanced.sh
```

Then access the chatbot at http://localhost:5000

## Detailed Integration Steps

### 1. Frontend Setup

The integration script will:
- Clone the GitHub repository
- Build the React frontend
- Copy the build files to the local directory

### 2. Backend Integration

The enhanced adapter will:
- Initialize all backend components
- Map API endpoints for compatibility
- Serve the frontend files
- Provide clickable citations with source viewer

## Adapter Features

### Enhanced Adapter (Recommended)

- Full RAG implementation with hybrid search
- Clickable document citations with source viewer
- Excel-specific references with sheet/row information
- Robust error handling and component compatibility

### Other Adapters

Each adapter provides different levels of functionality:

- **final_adapter.py**: Full functionality with robust component compatibility
- **complete_adapter.py**: Complete backend integration
- **stream_fixed_adapter.py**: Fixed streaming support
- **flexible_adapter.py**: Port-flexible implementation
- **adapter_step1_fixed.py**: Basic adapter with simple RAG

## Port Configuration

By default, adapters try to use port 5000 to match the frontend's expectations.
You can specify a different port with:

```bash
./run_enhanced.sh 8090
```

## Troubleshooting

### Common Issues

- **"Error talking to server"**: This usually indicates a port mismatch or CORS issue.
  - Solution: Use port 5000 or try the enhanced_adapter which fixes these issues.

- **Missing Frontend Files**: If the frontend isn't properly built.
  - Solution: Run `./simple_integration.sh` again.

- **[object Object] in Responses**: Indicates a streaming format issue.
  - Solution: Use enhanced_adapter.py which correctly formats responses.

### Checking Component Status

The adapters provide a health check endpoint at `/api/health` that shows the status
of each backend component.

## Acknowledgements

This integration combines:
- Local enhanced backend with robust RAG capabilities
- Modern GitHub frontend UI

## Contact

For questions, report issues at the GitHub repository.