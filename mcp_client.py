"""
Beats-to-Prose MCP Client
This module implements a Model Context Protocol (MCP) client for the Beats-to-Prose project.
It provides a simple interface to interact with the MCP server.
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable

# Import MCP SDK
from mcp import (
    MCPClient,
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPCapability
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BeatsToProseMCPClient(MCPClient):
    """MCP Client implementation for Beats-to-Prose"""
    
    def __init__(self, server_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize the MCP client
        
        Args:
            server_url: URL of the MCP server
            api_key: API key for authentication
        """
        # Initialize the MCP client
        super().__init__(
            server_url=server_url,
            api_key=api_key
        )
        
        # Store client capabilities
        self.capabilities = {}
    
    async def connect(self) -> Dict[str, Any]:
        """
        Connect to the MCP server and initialize client capabilities
        
        Returns:
            Dict containing server capabilities
        """
        # Define client capabilities
        client_capabilities = {
            "story_generation": True,
            "text_analysis": True,
            "style_evaluation": True
        }
        
        # Initialize connection
        response = await self.initialize(client_capabilities)
        
        # Store server capabilities
        self.capabilities = response.data.get("server_capabilities", {})
        
        return self.capabilities
    
    async def generate_story(
        self, 
        beats: List[str], 
        metadata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> str:
        """
        Generate a story from beats with optional metadata and analysis
        
        Args:
            beats: List of story beats
            metadata: Optional metadata for story generation
            options: Optional generation options
            callback: Optional callback function for status updates
            
        Returns:
            Story ID for tracking generation progress
        """
        # Check if story generation is enabled
        if not self.capabilities.get("story_generation", {}).get("enabled", False):
            raise MCPError(
                code="FEATURE_DISABLED",
                message="Story generation is not enabled on the server"
            )
        
        # Prepare request data
        data = {
            "beats": beats,
            "metadata": metadata or {},
            "options": options or {}
        }
        
        # Send request
        response = await self.request(
            endpoint="/story/generate",
            method="POST",
            data=data
        )
        
        # Extract story ID
        story_id = response.data.get("id")
        
        # Start polling for status updates if callback is provided
        if callback and story_id:
            asyncio.create_task(self._poll_story_status(story_id, callback))
        
        return story_id
    
    async def get_story_status(self, story_id: str) -> Dict[str, Any]:
        """
        Get the status of a story generation
        
        Args:
            story_id: ID of the story to check
            
        Returns:
            Dict containing story status
        """
        # Send request
        response = await self.request(
            endpoint=f"/story/{story_id}",
            method="GET"
        )
        
        return response.data
    
    async def analyze_text(
        self, 
        text: str, 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze text using available tools
        
        Args:
            text: Text to analyze
            options: Optional analysis options
            
        Returns:
            Dict containing analysis results
        """
        # Check if text analysis is enabled
        if not self.capabilities.get("text_analysis", {}).get("enabled", False):
            raise MCPError(
                code="FEATURE_DISABLED",
                message="Text analysis is not enabled on the server"
            )
        
        # Prepare request data
        data = {
            "text": text,
            "options": options or {}
        }
        
        # Send request
        response = await self.request(
            endpoint="/text/analyze",
            method="POST",
            data=data
        )
        
        return response.data
    
    async def evaluate_style(
        self, 
        generated_text: str, 
        style_reference: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate generated text against a style reference
        
        Args:
            generated_text: Text to evaluate
            style_reference: Reference style text
            options: Optional evaluation options
            
        Returns:
            Dict containing evaluation results
        """
        # Check if style evaluation is enabled
        if not self.capabilities.get("style_evaluation", {}).get("enabled", False):
            raise MCPError(
                code="FEATURE_DISABLED",
                message="Style evaluation is not enabled on the server"
            )
        
        # Prepare request data
        data = {
            "generated_text": generated_text,
            "style_reference": style_reference,
            "options": options or {}
        }
        
        # Send request
        response = await self.request(
            endpoint="/style/evaluate",
            method="POST",
            data=data
        )
        
        return response.data
    
    async def _poll_story_status(
        self, 
        story_id: str, 
        callback: Callable[[Dict[str, Any]], Awaitable[None]],
        interval: float = 1.0
    ):
        """
        Poll for story status updates
        
        Args:
            story_id: ID of the story to poll
            callback: Callback function for status updates
            interval: Polling interval in seconds
        """
        try:
            while True:
                # Get story status
                status = await self.get_story_status(story_id)
                
                # Call callback function
                await callback(status)
                
                # Check if generation is complete
                if status.get("status") in ["completed", "failed"]:
                    break
                
                # Wait before polling again
                await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Error polling story status: {str(e)}")

# Example usage
async def main():
    # Initialize client
    client = BeatsToProseMCPClient(
        server_url="http://localhost:8000",
        api_key="your_api_key"  # Replace with your API key
    )
    
    # Connect to server
    capabilities = await client.connect()
    logger.info(f"Connected to server with capabilities: {capabilities}")
    
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
        logger.info(f"Story status: {status.get('status')}, Progress: {status.get('progress', 0)}")
    
    # Start story generation
    story_id = await client.generate_story(beats, metadata, callback=status_callback)
    logger.info(f"Started story generation with ID: {story_id}")
    
    # Wait for generation to complete
    while True:
        status = await client.get_story_status(story_id)
        if status.get("status") in ["completed", "failed"]:
            break
        await asyncio.sleep(1)
    
    # Check result
    if status.get("status") == "completed":
        result = status.get("result", {})
        logger.info(f"Story generated successfully: {result.get('prose', '')[:100]}...")
    else:
        logger.error(f"Story generation failed: {status.get('error')}")
    
    # Analyze text
    analysis = await client.analyze_text("The young wizard cast a spell.")
    logger.info(f"Text analysis: {analysis}")
    
    # Evaluate style
    style_reference = "The old wizard's eyes twinkled with ancient wisdom."
    evaluation = await client.evaluate_style(
        "The young wizard's eyes sparkled with newfound power.",
        style_reference
    )
    logger.info(f"Style evaluation: {evaluation}")

# Run the example
if __name__ == "__main__":
    asyncio.run(main()) 