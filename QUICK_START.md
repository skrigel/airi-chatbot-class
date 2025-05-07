# Quick Start Guide for AIRI Chatbot

This guide will get you up and running with the AIRI Chatbot in just two steps.

## Complete Setup (Just Two Commands!)

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/airi-chatbot.git
cd airi-chatbot

# Run the setup script (does everything automatically)
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Install all dependencies
2. Build the frontend
3. Build the vector database (tries both standard and fallback methods)
4. Ask if you want to start the web interface immediately

When prompted "Do you want to start the chatbot web interface now?", press 'y' to launch the chatbot in your web browser right away.

That's it! The chatbot will be available at http://localhost:5000 (or another port if 5000 is unavailable).

## If You Ever Need to Rebuild the Database

If the chatbot isn't finding specific information as expected, you can rebuild the database:

```bash
# Option 1: Standard vector database (uses embeddings)
./rebuild_database.sh

# Option 2: Simple keyword-based database (no embeddings)
./run_simple_rebuild.sh
```

## Troubleshooting

1. **Missing dependencies**: 
   ```bash
   ./install_deps.sh
   ```

2. **Port conflicts**:
   The app automatically finds an available port, or specify one:
   ```bash
   ./run.sh 8080
   ```

3. **Excel file processing issues**:
   Make sure openpyxl is installed:
   ```bash
   pip install --user openpyxl
   ```

4. **Frontend issues**:
   If the frontend isn't loading correctly:
   ```bash
   ./setup.sh
   ```

## Example Questions to Test

- "What are the main types of AI risks?"
- "How might AI affect employment?"
- "What concerns exist regarding bias in AI systems?"
- "What are the risks of AI in healthcare?"
- "What are concerns about harmful language generation in AI?"
- "What might happen if self-driving cars get taken over?"