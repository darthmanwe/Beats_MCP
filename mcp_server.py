"""
Beats-to-Prose MCP Server
This module implements a Model Context Protocol (MCP) server for the Beats-to-Prose project.
It exposes the functionality of the beats_to_prose_solution.py module through the MCP protocol.
"""

import json
import logging
import os
import asyncio
import argparse
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

# Import MCP SDK
from mcp import (
    MCPServer, 
    MCPRequest, 
    MCPResponse, 
    MCPError, 
    MCPCapability,
    MCPAuthentication,
    MCPModel
)

# Import our existing models and functions
from beats_to_prose_solution import (
    StoryMetadata, ProseResponse, BeatsToProseRequest, 
    process_test_beats_part2, analyze_text_with_spacy,
    RAGConfig, SpacyConfig, Character, Setting, TextAnalysis
)

# Import LEPOR evaluator
from lepor_evaluator import LEPOREvaluator

# Import configuration
from mcp_config import load_config, ServerConfig, LLMConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server implementation
class BeatsToProseMCPServer(MCPServer):
    """MCP Server implementation for Beats-to-Prose"""
    
    def __init__(self, config: Optional[ServerConfig] = None):
        # Load configuration
        self.config = config or load_config()
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Initialize the MCP server
        super().__init__(
            name="beats-to-prose",
            version="1.0.0",
            description="MCP Server for generating prose from story beats with advanced analysis",
            protocol_version="1.0.0",
            authentication=MCPAuthentication(
                type="api_key",
                required=True
            )
        )
        
        # In-memory store for story generations
        self.story_generations = {}
        
        # Initialize LEPOR evaluator if enabled
        self.lepor_evaluator = LEPOREvaluator() if self.config.enable_lepor else None
        
        # Register capabilities
        self._register_capabilities()
        
        # Register routes
        self._register_routes()
    
    def _register_capabilities(self):
        """Register MCP capabilities"""
        
        # Story generation capability
        self.register_capability(
            MCPCapability(
                name="story_generation",
                description="Generate prose from story beats",
                enabled=True,
                features=["prose_generation", "metadata_analysis", "text_analysis", "rag_enhancement"]
            )
        )
        
        # Text analysis capability
        self.register_capability(
            MCPCapability(
                name="text_analysis",
                description="Analyze text using available tools",
                enabled=True,
                features=["entity_extraction", "sentiment_analysis", "readability_analysis", "linguistic_features"]
            )
        )
        
        # RAG enhancement capability
        self.register_capability(
            MCPCapability(
                name="rag_enhancement",
                description="Enhance story generation with RAG",
                enabled=self.config.enable_rag,
                features=["similar_examples", "style_matching", "genre_matching"]
            )
        )
        
        # Style evaluation capability
        self.register_capability(
            MCPCapability(
                name="style_evaluation",
                description="Evaluate generated text against a style reference",
                enabled=self.config.enable_lepor,
                features=["lepor_evaluation", "style_feature_analysis", "style_comparison"]
            )
        )
        
        # LLM integration capability
        self.register_capability(
            MCPCapability(
                name="llm_integration",
                description="Integration with LLM providers",
                enabled=True,
                provider=self.config.llm.provider,
                model=self.config.llm.model
            )
        )
    
    def _register_routes(self):
        """Register MCP routes"""
        
        # MCP story generation endpoint
        @self.route("/story/generate")
        async def generate_story(request: MCPRequest) -> MCPResponse:
            """Generate a story from beats with optional metadata and analysis"""
            logger.info("MCP story generation request received")
            
            # Extract request parameters
            beats = request.data.get("beats", [])
            metadata = request.data.get("metadata", {})
            options = request.data.get("options", {})
            
            # Validate input
            if not beats or not isinstance(beats, list):
                return MCPError(
                    code="INVALID_INPUT",
                    message="Invalid beats input. Must be a non-empty list."
                )
            
            # Create a new story ID and status
            story_id = str(uuid.uuid4())
            status = {
                "id": story_id,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "progress": 0.0
            }
            
            # Store the status
            self.story_generations[story_id] = status
            
            # Convert metadata to StoryMetadata if provided
            story_metadata = None
            if metadata:
                try:
                    # Extract characters
                    characters = None
                    if "characters" in metadata:
                        characters = [
                            Character(
                                name=char.get("name", ""),
                                role=char.get("role", ""),
                                description=char.get("description", "")
                            )
                            for char in metadata["characters"]
                        ]
                    
                    # Extract setting
                    setting = None
                    if "setting" in metadata:
                        setting = Setting(
                            location=metadata["setting"].get("location", "Unknown"),
                            time_period=metadata["setting"].get("time_period", ""),
                            atmosphere=metadata["setting"].get("atmosphere", "")
                        )
                    
                    # Create StoryMetadata
                    story_metadata = StoryMetadata(
                        characters=characters,
                        setting=setting,
                        genre=metadata.get("genre", ""),
                        style=metadata.get("style", "")
                    )
                except Exception as e:
                    logger.warning(f"Error creating StoryMetadata: {str(e)}")
            
            # Extract options
            use_rag = options.get("use_rag", self.config.enable_rag)
            use_spacy = options.get("use_spacy", self.config.enable_spacy)
            
            # Start the background task
            asyncio.create_task(
                self._generate_story_task, 
                story_id, 
                beats, 
                story_metadata, 
                use_rag, 
                use_spacy
            )
            
            return MCPResponse(data=status)
        
        # MCP story status endpoint
        @self.route("/story/{story_id}")
        async def get_story_status(request: MCPRequest) -> MCPResponse:
            """Get the status or result of a story generation"""
            story_id = request.params.get("story_id")
            
            if not story_id or story_id not in self.story_generations:
                return MCPError(
                    code="NOT_FOUND",
                    message="Story not found"
                )
            
            return MCPResponse(data=self.story_generations[story_id])
        
        # MCP text analysis endpoint
        @self.route("/text/analyze")
        async def analyze_text(request: MCPRequest) -> MCPResponse:
            """Analyze existing prose using available tools"""
            logger.info("MCP text analysis request received")
            
            # Extract request parameters
            text = request.data.get("text", "")
            options = request.data.get("options", {})
            
            # Validate input
            if not text:
                return MCPError(
                    code="INVALID_INPUT",
                    message="Invalid text input. Must be a non-empty string."
                )
            
            # Extract options
            use_spacy = options.get("use_spacy", self.config.enable_spacy)
            
            # Configure spaCy if requested
            spacy_config = None
            if use_spacy:
                spacy_config = SpacyConfig(
                    enabled=True,
                    model_name="en_core_web_sm",
                    analyze_entities=True,
                    analyze_sentiment=True,
                    analyze_readability=True,
                    analyze_linguistic_features=True
                )
            
            # Perform analysis
            analysis = {}
            if use_spacy and spacy_config:
                spacy_analysis = await asyncio.to_thread(
                    analyze_text_with_spacy,
                    text,
                    spacy_config
                )
                if spacy_analysis:
                    analysis["spacy_analysis"] = spacy_analysis.dict()
            
            return MCPResponse(data=analysis)
        
        # MCP style evaluation endpoint
        @self.route("/style/evaluate")
        async def evaluate_style(request: MCPRequest) -> MCPResponse:
            """Evaluate generated text against a style reference using LEPOR"""
            logger.info("MCP style evaluation request received")
            
            # Check if LEPOR is enabled
            if not self.config.enable_lepor or not self.lepor_evaluator:
                return MCPError(
                    code="FEATURE_DISABLED",
                    message="LEPOR evaluation is not enabled"
                )
            
            # Extract request parameters
            generated_text = request.data.get("generated_text", "")
            style_reference = request.data.get("style_reference", "")
            options = request.data.get("options", {})
            
            # Validate input
            if not generated_text:
                return MCPError(
                    code="INVALID_INPUT",
                    message="Invalid generated text. Must be a non-empty string."
                )
            if not style_reference:
                return MCPError(
                    code="INVALID_INPUT",
                    message="Invalid style reference. Must be a non-empty string."
                )
            
            # Extract options
            evaluation_type = options.get("evaluation_type", "comprehensive")
            
            # Perform evaluation
            try:
                if evaluation_type == "lepor_only":
                    # LEPOR scores only
                    result = await asyncio.to_thread(
                        self.lepor_evaluator.evaluate,
                        generated_text,
                        style_reference
                    )
                elif evaluation_type == "style_features_only":
                    # Style features comparison only
                    result = await asyncio.to_thread(
                        self.lepor_evaluator.compare_style_features,
                        generated_text,
                        style_reference
                    )
                else:
                    # Comprehensive evaluation (default)
                    result = await asyncio.to_thread(
                        self.lepor_evaluator.evaluate_with_style,
                        generated_text,
                        style_reference
                    )
                
                return MCPResponse(data=result)
            except Exception as e:
                logger.error(f"Error in style evaluation: {str(e)}")
                return MCPError(
                    code="INTERNAL_ERROR",
                    message=f"Error in style evaluation: {str(e)}"
                )
    
    async def _generate_story_task(
        self, 
        story_id: str, 
        beats: List[str], 
        metadata: Optional[StoryMetadata] = None,
        use_rag: bool = False,
        use_spacy: bool = False
    ):
        """Background task for story generation"""
        try:
            # Update status
            self.story_generations[story_id]["status"] = "processing"
            
            # Generate output filename
            output_file = f"story_{story_id}.txt"
            
            # Process the story beats
            result = await asyncio.to_thread(
                process_test_beats_part2,
                client=None,  # Will use global client
                beats_input=beats,
                output_file=output_file,
                include_rag=use_rag,
                include_spacy=use_spacy
            )
            
            # Update status with success
            self.story_generations[story_id]["status"] = "completed"
            self.story_generations[story_id]["completed_at"] = datetime.now().isoformat()
            self.story_generations[story_id]["result"] = result.dict()
            self.story_generations[story_id]["progress"] = 1.0
            
        except Exception as e:
            # Update status with failure
            self.story_generations[story_id]["status"] = "failed"
            self.story_generations[story_id]["error"] = str(e)
            self.story_generations[story_id]["completed_at"] = datetime.now().isoformat()
            logger.error(f"Error generating story: {str(e)}")

def create_server(config_path: Optional[str] = None) -> BeatsToProseMCPServer:
    """
    Create an MCP server with the given configuration
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        BeatsToProseMCPServer instance
    """
    config = load_config(config_path)
    return BeatsToProseMCPServer(config)

# Create MCP server instance
mcp_server = create_server()

# Export the FastAPI app
app = mcp_server.app

# Run the server if this file is executed directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beats-to-Prose MCP Server")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--host", type=str, help="Host to run the server on")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    
    # Create server
    server = BeatsToProseMCPServer(config)
    
    # Run server
    import uvicorn
    uvicorn.run(server.app, host=config.host, port=config.port) 