"""
Beats-to-Prose MCP Client
This module implements a Model Context Protocol (MCP) client for the Beats-to-Prose project.
It provides a simple interface to interact with the MCP server.
"""

import json
import logging
import asyncio
import aiohttp
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPClient:
    """MCP Client implementation for Beats-to-Prose"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """Initialize the MCP client
        
        Args:
            server_url: URL of the MCP server
        """
        self.server_url = server_url
        self.session = None
        self.capabilities = {}
        self.initialized = False
    
    async def connect(self):
        """Connect to the MCP server and initialize the connection"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        # Define client capabilities
        client_capabilities = {
            "story_generation": {
                "enabled": True,
                "features": ["prose_generation", "metadata_analysis"]
            },
            "text_analysis": {
                "enabled": True,
                "features": ["entity_extraction", "sentiment_analysis"]
            },
            "style_evaluation": {
                "enabled": True,
                "features": ["lepor_evaluation", "style_feature_analysis"]
            }
        }
        
        # Initialize MCP connection
        try:
            async with self.session.post(
                f"{self.server_url}/mcp/initialize",
                json={"capabilities": client_capabilities}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.capabilities = result.get("server_capabilities", {})
                    self.initialized = True
                    logger.info("MCP connection initialized successfully")
                    return True
                else:
                    logger.error(f"Failed to initialize MCP connection: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error connecting to MCP server: {str(e)}")
            return False
    
    async def close(self):
        """Close the MCP client connection"""
        if self.session:
            await self.session.close()
            self.session = None
            self.initialized = False
            logger.info("MCP connection closed")
    
    async def generate_story(
        self, 
        beats: List[str], 
        metadata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Generate a story from beats with optional metadata and analysis
        
        Args:
            beats: List of story beats
            metadata: Optional metadata for the story
            options: Optional generation options
            callback: Optional callback function to receive status updates
            
        Returns:
            Dict containing the story generation result
        """
        if not self.initialized:
            await self.connect()
        
        if not self.session:
            raise Exception("MCP client not connected")
        
        # Prepare request
        request = {
            "beats": beats,
            "metadata": metadata or {},
            "options": options or {}
        }
        
        # Send request
        try:
            async with self.session.post(
                f"{self.server_url}/mcp/story/generate",
                json=request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    story_id = result.get("id")
                    
                    if not story_id:
                        raise Exception("No story ID returned from server")
                    
                    # Poll for status updates
                    while True:
                        status = await self.get_story_status(story_id)
                        
                        # Call callback if provided
                        if callback:
                            callback(status)
                        
                        # Check if generation is complete
                        if status.get("status") in ["completed", "failed"]:
                            break
                        
                        # Wait before polling again
                        await asyncio.sleep(1)
                    
                    return status
                else:
                    logger.error(f"Failed to generate story: {response.status}")
                    return {"status": "failed", "error": f"Server returned status code {response.status}"}
        except Exception as e:
            logger.error(f"Error generating story: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def get_story_status(self, story_id: str) -> Dict[str, Any]:
        """Get the status of a story generation
        
        Args:
            story_id: ID of the story generation
            
        Returns:
            Dict containing the story generation status
        """
        if not self.initialized:
            await self.connect()
        
        if not self.session:
            raise Exception("MCP client not connected")
        
        try:
            async with self.session.get(
                f"{self.server_url}/mcp/story/{story_id}"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get story status: {response.status}")
                    return {"status": "failed", "error": f"Server returned status code {response.status}"}
        except Exception as e:
            logger.error(f"Error getting story status: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze text using available tools
        
        Args:
            text: Text to analyze
            options: Optional analysis options
            
        Returns:
            Dict containing the analysis results
        """
        if not self.initialized:
            await self.connect()
        
        if not self.session:
            raise Exception("MCP client not connected")
        
        # Prepare request
        request = {
            "text": text,
            "options": options or {}
        }
        
        try:
            async with self.session.post(
                f"{self.server_url}/mcp/text/analyze",
                json=request
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to analyze text: {response.status}")
                    return {"status": "failed", "error": f"Server returned status code {response.status}"}
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def evaluate_style(
        self, 
        generated_text: str, 
        style_reference: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate generated text against a style reference using LEPOR
        
        Args:
            generated_text: Generated text to evaluate
            style_reference: Reference style text
            options: Optional evaluation options
            
        Returns:
            Dict containing the evaluation results
        """
        if not self.initialized:
            await self.connect()
        
        if not self.session:
            raise Exception("MCP client not connected")
        
        # Prepare request
        request = {
            "generated_text": generated_text,
            "style_reference": style_reference,
            "options": options or {}
        }
        
        try:
            async with self.session.post(
                f"{self.server_url}/mcp/style/evaluate",
                json=request
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to evaluate style: {response.status}")
                    return {"status": "failed", "error": f"Server returned status code {response.status}"}
        except Exception as e:
            logger.error(f"Error evaluating style: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the MCP server
        
        Returns:
            Dict containing the health check result
        """
        if not self.initialized:
            await self.connect()
        
        if not self.session:
            raise Exception("MCP client not connected")
        
        try:
            async with self.session.get(
                f"{self.server_url}/mcp/health"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Health check failed: {response.status}")
                    return {"status": "unhealthy", "error": f"Server returned status code {response.status}"}
        except Exception as e:
            logger.error(f"Error checking health: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

# Example usage
async def main():
    """Example usage of the MCP client"""
    client = MCPClient()
    
    try:
        # Connect to the server
        if await client.connect():
            logger.info("Connected to MCP server")
            
            # Check server health
            health = await client.health_check()
            logger.info(f"Server health: {health}")
            
            # Generate a story
            beats = [
                "A young wizard discovers they have magical powers",
                "They are invited to a magical school",
                "They face a dark force threatening the magical world"
            ]
            
            metadata = {
                "characters": [
                    {
                        "name": "Alex",
                        "role": "protagonist",
                        "description": "A young person discovering their magical abilities"
                    }
                ],
                "setting": {
                    "location": "Magical world",
                    "time_period": "Present day",
                    "atmosphere": "Mysterious and enchanting"
                },
                "genre": "Fantasy",
                "style": "Young adult"
            }
            
            options = {
                "use_rag": True,
                "use_spacy": True
            }
            
            # Define a callback function to receive status updates
            def status_callback(status):
                logger.info(f"Story generation status: {status.get('status')} - Progress: {status.get('progress', 0)}")
            
            # Generate the story
            result = await client.generate_story(beats, metadata, options, status_callback)
            
            # Check the result
            if result.get("status") == "completed":
                logger.info("Story generation completed successfully")
                story_result = result.get("result", {})
                prose = story_result.get("prose", "")
                logger.info(f"Generated prose ({len(prose.split())} words):")
                logger.info(prose[:500] + "..." if len(prose) > 500 else prose)
                
                # Evaluate the style
                style_reference = """
                The ancient castle stood atop the misty mountain, its weathered stones telling tales of centuries past. 
                Within its walls, young apprentices practiced their craft, their eyes alight with wonder and determination. 
                The air was thick with the scent of old books and the crackle of magical energy, as if the very fabric of reality 
                bent to the will of those who knew its secrets.
                """
                
                style_evaluation = await client.evaluate_style(prose, style_reference)
                logger.info("Style evaluation results:")
                logger.info(json.dumps(style_evaluation, indent=2))
            else:
                logger.error(f"Story generation failed: {result.get('error')}")
            
            # Analyze some text
            text = "The magical school stood atop a misty mountain, its ancient spires reaching toward the clouds."
            analysis = await client.analyze_text(text, {"use_spacy": True})
            logger.info(f"Text analysis: {json.dumps(analysis, indent=2)}")
        
    finally:
        # Close the connection
        await client.close()

# Run the example if this file is executed directly
if __name__ == "__main__":
    asyncio.run(main()) 