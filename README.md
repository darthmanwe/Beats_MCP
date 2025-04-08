# Beats-to-Prose MCP Server

A Model Context Protocol (MCP) server implementation for the Beats-to-Prose project, which converts story beats into prose using AI.

## Features

- RESTful API endpoints for story generation and text analysis
- Asynchronous processing for long-running tasks
- Status tracking for story generation
- Text analysis capabilities using spaCy
- LEPOR-based style evaluation
- Configurable LLM provider integration
- Health monitoring
- Client implementation for easy integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/beats-to-prose.git
cd beats-to-prose
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
# Optional: For other LLM providers
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key
```

## Configuration

The MCP server can be configured using a JSON configuration file or environment variables.

### Configuration File

Create a `config.json` file in the root directory:

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your_api_key",
    "temperature": 0.7,
    "max_tokens": 4000
  },
  "enable_rag": true,
  "enable_spacy": true,
  "enable_lepor": true,
  "log_level": "INFO"
}
```

### Environment Variables

You can also configure the server using environment variables:

```
MCP_HOST=0.0.0.0
MCP_PORT=8000
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

Basic usage:
```bash
python mcp_server.py
```

With configuration file:
```bash
python mcp_server.py --config config.json
```

With command-line overrides:
```bash
python mcp_server.py --config config.json --host 127.0.0.1 --port 8080
```

The server will start on the specified host and port.

### Using the Client

```python
from mcp_client import MCPClient

async def main():
    # Initialize client
    client = MCPClient()
    
    # Connect to server
    await client.connect()
    
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
    
    # Start story generation
    story_id = await client.generate_story(beats, metadata)
    
    # Check status
    status = await client.get_story_status(story_id)
    print(f"Story status: {status}")
    
    # Analyze text
    analysis = await client.analyze_text("The young wizard cast a spell.")
    print(f"Text analysis: {analysis}")
    
    # Evaluate style
    style_reference = "The old wizard's eyes twinkled with ancient wisdom."
    evaluation = await client.evaluate_style(
        "The young wizard'\''s eyes sparkled with newfound power.",
        style_reference
    )
    print(f"Style evaluation: {evaluation}")
    
    # Close connection
    await client.close()

# Run the example
import asyncio
asyncio.run(main())
```

## API Endpoints

### MCP Endpoints

- `POST /mcp/initialize`: Initialize MCP connection
- `POST /mcp/story/generate`: Generate a story from beats
- `GET /mcp/story/{story_id}`: Get story generation status
- `POST /mcp/text/analyze`: Analyze text using available tools
- `POST /mcp/style/evaluate`: Evaluate generated text against a style reference
- `GET /mcp/health`: Check server health

### Example API Requests

#### Initialize MCP Connection

```bash
curl -X POST http://localhost:8000/mcp/initialize \
  -H "Content-Type: application/json" \
  -d '{"capabilities": {"story_generation": true}}'
```

#### Generate a Story

```bash
curl -X POST http://localhost:8000/mcp/story/generate \
  -H "Content-Type: application/json" \
  -d '{
    "beats": [
      "A young wizard discovers his magical powers",
      "He must face a dark wizard who threatens his school"
    ],
    "metadata": {
      "genre": "fantasy",
      "target_audience": "young adult",
      "tone": "adventurous"
    },
    "options": {
      "use_rag": true,
      "use_spacy": true
    }
  }'
```

#### Check Story Status

```bash
curl -X GET http://localhost:8000/mcp/story/123e4567-e89b-12d3-a456-426614174000
```

#### Analyze Text

```bash
curl -X POST http://localhost:8000/mcp/text/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The young wizard cast a spell.",
    "options": {
      "use_spacy": true
    }
  }'
```

#### Evaluate Style

```bash
curl -X POST http://localhost:8000/mcp/style/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "generated_text": "The young wizard'\''s eyes sparkled with newfound power.",
    "style_reference": "The old wizard'\''s eyes twinkled with ancient wisdom.",
    "options": {
      "evaluation_type": "comprehensive"
    }
  }'
```

## Changelog

### v1.0.0
- Initial release with basic story generation functionality
- Added MCP server implementation
- Added MCP client implementation
- Added text analysis capabilities using spaCy
- Added LEPOR-based style evaluation
- Added configurable LLM provider integration
- Added configuration system with file and environment variable support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 