import os
import subprocess
import sys
import tempfile
import time
import requests
from typing import Dict, Any, List, Optional
from shared.logging_config import get_logger

logger = get_logger(__name__)

class ToolsService:
    def __init__(self):
        self.google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    def run_python_code(self, code: str) -> Dict[str, Any]:
        """
        Executes Python code in a subprocess and returns the output.
        NOTE: In production, this SHOULD be in a gVisor or Docker sandbox.
        """
        logger.info(f"[TOOLS] Executing Python code: {code[:100]}...")
        
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as f:
            f.write(code)
            temp_file_path = f.name

        try:
            # Execute with a 5 second timeout
            start_time = time.time()
            process = subprocess.Popen(
                [sys.executable, temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=5)
            execution_time = time.time() - start_time
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": process.returncode,
                "execution_time": execution_time
            }
        except subprocess.TimeoutExpired:
            process.kill()
            return {
                "stdout": "",
                "stderr": "Error: Execution timed out (5s limit)",
                "exit_code": -1,
                "execution_time": 5.0
            }
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "exit_code": -1,
                "execution_time": 0
            }
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """
        Simulated knowledge base search.
        In a real app, this would use a Vector DB or Google Search API.
        """
        logger.info(f"[TOOLS] Searching knowledge base for: {query}")
        
        # For demo purposes, we'll use a mocked response if keys are missing
        if not self.google_search_api_key:
            return [
                {
                    "title": f"Result for {query}",
                    "snippet": f"This is a simulated result for your search about {query}. In production, this would be a real search result.",
                    "link": "https://en.wikipedia.org/wiki/Learning"
                }
            ]

        # Actual Google Search integration if keys exist
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_search_api_key,
                "cx": self.google_search_engine_id,
                "q": query,
                "num": 3
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "link": item.get("link")
                })
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return [{"error": str(e)}]

tools_service = ToolsService()
