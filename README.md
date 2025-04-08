# Beats-to-Prose MCP

A Model Context Protocol (MCP) server for converting story beats into prose using AI. This project implements the official Model Context Protocol specification for AI model integration.

## Features

- RESTful API endpoints for story generation and text analysis
- Asynchronous processing with status tracking
- Text analysis capabilities using spaCy
- LEPOR-based style evaluation
- Configurable LLM provider integration
- Client implementation for easy integration
- Official Model Context Protocol (MCP) compliance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/darthmanwe/Beats_MCP.git
cd Beats_MCP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file
cp .env.example .env

# Edit the .env file with your API keys and configuration
```

## Configuration

The server can be configured using a JSON configuration file or environment variables:

### Configuration File

Create a `config.json` file:

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "log_level": "INFO",
  "enable_rag": false,
  "enable_spacy": true,
  "enable_lepor": true,
  "llm": {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "api_key": "your_api_key",
    "temperature": 0.7,
    "max_tokens": 4000
  }
}
```

### Environment Variables

```bash
# Server settings
MCP_HOST=0.0.0.0
MCP_PORT=8000
MCP_LOG_LEVEL=INFO

# Feature toggles
MCP_ENABLE_RAG=true
MCP_ENABLE_SPACY=true
MCP_ENABLE_LEPOR=true

# LLM settings
MCP_LLM_PROVIDER=openai
MCP_LLM_MODEL=gpt-4
MCP_LLM_API_KEY=your_api_key
MCP_LLM_TEMPERATURE=0.7
MCP_LLM_MAX_TOKENS=4000
MCP_ENABLE_RAG=true
MCP_ENABLE_SPACY=true
MCP_ENABLE_LEPOR=true
MCP_LOG_LEVEL=INFO
```

## Usage

### Starting the Server

```bash
# Using default configuration
python mcp_server.py

# Using a configuration file
python mcp_server.py --config config.json

# Using command line arguments
python mcp_server.py --host 0.0.0.0 --port 8000
```

### Using the Client

```python
import asyncio
from mcp_client import BeatsToProseMCPClient

async def main():
    # Initialize client with API key
    client = BeatsToProseMCPClient(
        server_url="http://localhost:8000",
        api_key="your_api_key"
    )
    
    # Connect to server
    capabilities = await client.connect()
    print(f"Connected to server with capabilities: {capabilities}")
    
    # Generate a story
    beats = [
        "A young wizard discovers his magical powers",
        "He must face a dark wizard who threatens his school"
    ]
    
    metadata = {
        "genre": "fantasy",
        "target_audience": "young adult",
        "tone": "adventurous"
    }
    
    # Define status callback
    async def status_callback(status):
        print(f"Story status: {status.get('status')}, Progress: {status.get('progress', 0)}")
    
    # Start story generation
    story_id = await client.generate_story(beats, metadata, callback=status_callback)
    print(f"Started story generation with ID: {story_id}")
    
    # Wait for generation to complete
    while True:
        status = await client.get_story_status(story_id)
        if status.get("status") in ["completed", "failed"]:
            break
        await asyncio.sleep(1)
    
    # Check result
    if status.get("status") == "completed":
        result = status.get("result", {})
        print(f"Story generated successfully: {result.get('prose', '')[:100]}...")
    else:
        print(f"Story generation failed: {status.get('error')}")
    
    # Analyze text
    analysis = await client.analyze_text("The young wizard cast a spell.")
    print(f"Text analysis: {analysis}")
    
    # Evaluate style
    style_reference = "The old wizard's eyes twinkled with ancient wisdom."
    evaluation = await client.evaluate_style(
        "The young wizard's eyes sparkled with newfound power.",
        style_reference
    )
    print(f"Style evaluation: {evaluation}")

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

## API Endpoints

The server implements the following MCP endpoints:

### Story Generation

- `POST /story/generate` - Generate a story from beats
  ```json
  {
    "beats": ["beat1", "beat2", ...],
    "metadata": {
      "genre": "fantasy",
      "target_audience": "young adult",
      "tone": "adventurous"
    },
    "options": {
      "use_rag": true,
      "use_spacy": true
    }
  }
  ```

- `GET /story/{story_id}` - Get story generation status

### Text Analysis

- `POST /text/analyze` - Analyze text
  ```json
  {
    "text": "Text to analyze",
    "options": {
      "use_spacy": true
    }
  }
  ```

### Style Evaluation

- `POST /style/evaluate` - Evaluate text against a style reference
  ```json
  {
    "generated_text": "Text to evaluate",
    "style_reference": "Reference style text",
    "options": {
      "evaluation_type": "comprehensive"
    }
  }
  ```

## MCP Protocol

This implementation follows the official Model Context Protocol specification:

- **Initialization**: The client and server exchange capabilities during initialization
- **Authentication**: API key-based authentication is supported
- **Standardized Endpoints**: All endpoints follow the MCP specification
- **Error Handling**: Standardized error responses using MCP error codes
- **Response Format**: All responses follow the MCP response format

## Changelog

### v1.0.0

- Initial release with MCP server and client implementation
- Story generation from beats with metadata support
- Text analysis using spaCy
- LEPOR-based style evaluation
- Configurable LLM provider integration
- Official Model Context Protocol (MCP) compliance



## License

This project is licensed under the MIT License - see the LICENSE file for details. 