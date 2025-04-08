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
class MCPServer:
    """MCP Server implementation for Beats-to-Prose"""
    
    def __init__(self, config: Optional[ServerConfig] = None):
        # Load configuration
        self.config = config or load_config()
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        self.app = FastAPI(
            title="Beats-to-Prose MCP Server",
            description="MCP Server for generating prose from story beats with advanced analysis",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, replace with specific origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # In-memory store for story generations
        self.story_generations = {}
        
        # Initialize LEPOR evaluator if enabled
        self.lepor_evaluator = LEPOREvaluator() if self.config.enable_lepor else None
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register MCP routes"""
        
        # MCP initialization endpoint
        @self.app.post("/mcp/initialize")
        async def initialize_mcp(request: Dict[str, Any]):
            """Initialize MCP connection"""
            logger.info("MCP initialization request received")
            
            # Extract client capabilities
            client_capabilities = request.get("capabilities", {})
            
            # Define server capabilities
            server_capabilities = {
                "story_generation": {
                    "enabled": True,
                    "features": ["prose_generation", "metadata_analysis", "text_analysis", "rag_enhancement"]
                },
                "text_analysis": {
                    "enabled": True,
                    "features": ["entity_extraction", "sentiment_analysis", "readability_analysis", "linguistic_features"]
                },
                "rag_enhancement": {
                    "enabled": self.config.enable_rag,
                    "features": ["similar_examples", "style_matching", "genre_matching"]
                },
                "style_evaluation": {
                    "enabled": self.config.enable_lepor,
                    "features": ["lepor_evaluation", "style_feature_analysis", "style_comparison"]
                },
                "llm_integration": {
                    "enabled": True,
                    "provider": self.config.llm.provider,
                    "model": self.config.llm.model
                }
            }
            
            return {
                "status": "success",
                "server_capabilities": server_capabilities,
                "protocol_version": "1.0.0"
            }
        
        # MCP story generation endpoint
        @self.app.post("/mcp/story/generate")
        async def generate_story_mcp(request: Dict[str, Any], background_tasks: BackgroundTasks):
            """Generate a story from beats with optional metadata and analysis"""
            logger.info("MCP story generation request received")
            
            # Extract request parameters
            beats = request.get("beats", [])
            metadata = request.get("metadata", {})
            options = request.get("options", {})
            
            # Validate input
            if not beats or not isinstance(beats, list):
                raise HTTPException(status_code=400, detail="Invalid beats input. Must be a non-empty list.")
            
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
            background_tasks.add_task(
                self._generate_story_task, 
                story_id, 
                beats, 
                story_metadata, 
                use_rag, 
                use_spacy
            )
            
            return status
        
        # MCP story status endpoint
        @self.app.get("/mcp/story/{story_id}")
        async def get_story_status_mcp(story_id: str):
            """Get the status or result of a story generation"""
            if story_id not in self.story_generations:
                raise HTTPException(status_code=404, detail="Story not found")
            
            return self.story_generations[story_id]
        
        # MCP text analysis endpoint
        @self.app.post("/mcp/text/analyze")
        async def analyze_text_mcp(request: Dict[str, Any]):
            """Analyze existing prose using available tools"""
            logger.info("MCP text analysis request received")
            
            # Extract request parameters
            text = request.get("text", "")
            options = request.get("options", {})
            
            # Validate input
            if not text:
                raise HTTPException(status_code=400, detail="Invalid text input. Must be a non-empty string.")
            
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
            
            return analysis
        
        # MCP style evaluation endpoint
        @self.app.post("/mcp/style/evaluate")
        async def evaluate_style_mcp(request: Dict[str, Any]):
            """Evaluate generated text against a style reference using LEPOR"""
            logger.info("MCP style evaluation request received")
            
            # Check if LEPOR is enabled
            if not self.config.enable_lepor or not self.lepor_evaluator:
                raise HTTPException(status_code=400, detail="LEPOR evaluation is not enabled")
            
            # Extract request parameters
            generated_text = request.get("generated_text", "")
            style_reference = request.get("style_reference", "")
            options = request.get("options", {})
            
            # Validate input
            if not generated_text:
                raise HTTPException(status_code=400, detail="Invalid generated text. Must be a non-empty string.")
            if not style_reference:
                raise HTTPException(status_code=400, detail="Invalid style reference. Must be a non-empty string.")
            
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
                
                return result
            except Exception as e:
                logger.error(f"Error in style evaluation: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error in style evaluation: {str(e)}")
        
        # MCP health check endpoint
        @self.app.get("/mcp/health")
        async def health_check_mcp():
            """Health check endpoint"""
            return {"status": "healthy"}
    
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

def create_server(config_path: Optional[str] = None) -> MCPServer:
    """
    Create an MCP server with the given configuration
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        MCPServer instance
    """
    config = load_config(config_path)
    return MCPServer(config)

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
    server = MCPServer(config)
    
    # Run server
    import uvicorn
    uvicorn.run(server.app, host=config.host, port=config.port) 