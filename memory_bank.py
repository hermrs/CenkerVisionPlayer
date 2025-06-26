import os
import json
import subprocess
from typing import Dict, Any, Optional

class MemoryBank:
    def __init__(self, memory_bank_path: str = None):
        self.memory_bank_path = memory_bank_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "memory-bank"
        )
        
    def initialize(self, goal: str) -> Dict[str, Any]:
        """Yeni bir Memory Bank yapısı oluşturur."""
        result = self._call_mcp_tool("initialize_memory_bank", {
            "goal": goal,
            "location": self.memory_bank_path
        })
        return result

    def update_document(self, document_type: str, content: str) -> Dict[str, Any]:
        """Belirli bir dokümanı günceller."""
        result = self._call_mcp_tool("update_document", {
            "documentType": document_type,
            "content": content
        })
        return result

    def query(self, query: str) -> Dict[str, Any]:
        """Dokümanlarda arama yapar."""
        result = self._call_mcp_tool("query_memory_bank", {
            "query": query
        })
        return result

    def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP aracını çağırır."""
        try:
            # MCP sunucusuna istek gönder
            process = subprocess.Popen(
                ["node", os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    "memory-bank-mcp/dist/index.js")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # İsteği hazırla
            request = {
                "name": tool_name,
                "arguments": arguments
            }
            
            # İsteği gönder
            stdout, stderr = process.communicate(json.dumps(request))
            
            if process.returncode != 0:
                raise Exception(f"MCP tool error: {stderr}")
            
            return json.loads(stdout)
        except Exception as e:
            print(f"Error calling MCP tool: {e}")
            return {"error": str(e)} 