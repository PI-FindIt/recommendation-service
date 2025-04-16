from typing import List, Dict, Any
import aiohttp

class DataService:
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        # Implementar chamada para o serviço de usuários
        pass
        
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        # Implementar chamada para o serviço de usuários
        pass
        
    async def get_product_details(self, ean: str) -> Dict[str, Any]:
        # Implementar chamada para o serviço de produtos
        pass
