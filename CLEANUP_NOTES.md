# Project Cleanup Notes

This document explains the reorganization of the AIRI Chatbot project.

## Simplified File Structure

The project has been reorganized as follows:

### Core Files

- `airi_adapter.py` - The complete adapter combining all features
- `setup.sh` - Single script to set up the frontend and environment
- `run.sh` - Simple script to run the adapter

### Removed Unnecessary Files

The following intermediate implementations have been removed:

- Multiple adapter_*.py files with incremental features
- Various run_*.sh scripts for different adapters
- Separate integration scripts and patches

## Key Improvements

1. **Consolidated Adapter**:
   - Single, comprehensive implementation in `airi_adapter.py`
   - Well-documented with clear section headers
   - Handles all API compatibility and frontend integration

2. **Simplified Setup**:
   - One setup script `setup.sh` that handles:
     - GitHub repo cloning
     - Frontend building
     - Environment preparation
     - API URL patching

3. **Consistent Frontend Integration**:
   - Frontend directory renamed to "frontend" for clarity
   - Built-in document viewer for citations
   - Automated API URL patching

4. **Clean Codebase**:
   - Removed redundant files
   - Organized by function
   - Better error handling
   - Comprehensive documentation

## Migration Path

If you were using any of the previous adapter implementations:

1. Run `./setup.sh` to set up the new environment
2. Use `./run.sh` to start the unified adapter
3. Previous URLs and endpoints will continue to work

## Features Preserved

All features from the various adapter implementations have been preserved:

- Clickable citations with source viewer
- Domain-specific query handling
- Streaming responses
- Port flexibility
- Excel-specific references with sheet/row
- Full RAG implementation

## Additional Notes

The reorganization simplifies maintenance and future development by:

1. Having a single, well-structured codebase
2. Providing clear documentation
3. Using consistent naming and organization
4. Making setup and deployment straightforward