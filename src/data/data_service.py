import requests
from rich.console import Console

console = Console()

class DataService:
    def __init__(self, api_url: str = "http://localhost"):
        self.api_url = api_url
        self.console = Console()

    def _execute_query(self, query: str, variables: dict | None = None) -> dict:
        """Execute a GraphQL query and return the response"""
        try:
            response = requests.post(
                self.api_url,
                json={"query": query, "variables": variables or {}}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Error executing GraphQL query: {str(e)}[/bold red]")
            raise

    def get_user_data(self, user_id: str) -> dict:
        """
        Fetch user data including preferences and shopping history
        """
        query = """
        query GetUserData($userId: String!) {
            user(id: $userId) {
                _id
                first_name
                last_name
                preferences {
                    supermarketsLike
                    supermarketsDislike
                    brandsLike
                    brandsDislike
                    maxDistance
                    budget
                    pathType
                }
                supermarketLists {
                    _id
                    timestamp
                    status
                    products {
                        product {
                            ean
                            name
                            brandName
                            categoryName
                        }
                        supermarket {
                            id
                            name
                            price
                        }
                        quantity
                        status
                    }
                }
            }
        }
        """
        
        try:
            response = self._execute_query(query, {"userId": user_id})
            return response["data"]["user"]
        except Exception as e:
            console.print(f"[bold red]Error fetching user data: {str(e)}[/bold red]")
            raise

    def get_products_by_brand(self, brand_name: str) -> list[dict]:
        """
        Fetch products by brand name
        """
        query = """
        query GetProductsByBrand($filters: ProductFilter!) {
            products(filters: $filters) {
                ean
                name
                genericName
                brandName
                categoryName
                nutriScore
                supermarkets {
                    price
                    supermarket {
                        id
                        name
                    }
                }
            }
        }
        """
        
        variables = {
            "filters": {
                "brandName": {
                    "value": brand_name,
                    "op": "EQ"
                }
            }
        }
        
        try:
            response = self._execute_query(query, variables)
            return response["data"]["products"]
        except Exception as e:
            console.print(f"[bold red]Error fetching products by brand: {str(e)}[/bold red]")
            return []

    def get_products_by_category(self, category_name: str) -> list[dict]:
        """
        Fetch products by category name
        """
        query = """
        query GetProductsByCategory($filters: ProductFilter!) {
            products(filters: $filters) {
                ean
                name
                genericName
                brandName
                categoryName
                nutriScore
                supermarkets {
                    price
                    supermarket {
                        id
                        name
                    }
                }
            }
        }
        """
        
        variables = {
            "filters": {
                "categoryName": {
                    "value": category_name,
                    "op": "EQ"
                }
            }
        }
        
        try:
            response = self._execute_query(query, variables)
            return response["data"]["products"]
        except Exception as e:
            console.print(f"[bold red]Error fetching products by category: {str(e)}[/bold red]")
            return []

    def get_product_details(self, ean: str) -> dict | None:
        """
        Fetch detailed product information by EAN
        """
        query = """
        query GetProduct($ean: String!) {
            product(ean: $ean) {
                ean
                name
                genericName
                brandName
                categoryName
                nutriScore
                ingredients
                quantity
                unit
                keywords
                images
                nutrition
                supermarkets {
                    price
                    supermarket {
                        id
                        name
                        services
                        locations {
                            latitude
                            longitude
                        }
                    }
                }
            }
        }
        """
        
        try:
            response = self._execute_query(query, {"ean": ean})
            return response["data"]["product"]
        except Exception as e:
            console.print(f"[bold red]Error fetching product details: {str(e)}[/bold red]")
            return None

    def get_supermarket_details(self, supermarket_id: int) -> dict | None:
        """
        Fetch detailed supermarket information
        """
        query = """
        query GetSupermarket($id: Int!) {
            supermarket(id: $id) {
                id
                name
                image
                services
                description
                locations {
                    id
                    name
                    latitude
                    longitude
                }
            }
        }
        """
        
        try:
            response = self._execute_query(query, {"id": supermarket_id})
            return response["data"]["supermarket"]
        except Exception as e:
            console.print(f"[bold red]Error fetching supermarket details: {str(e)}[/bold red]")
            return None

    def get_user_shopping_history(self, user_id: str) -> list[dict]:
        """
        Fetch user's shopping history with detailed product information
        """
        query = """
        query GetUserShoppingHistory($userId: String!) {
            user(id: $userId) {
                supermarketLists {
                    _id
                    timestamp
                    status
                    products {
                        product {
                            ean
                            name
                            brandName
                            categoryName
                            nutriScore
                        }
                        supermarket {
                            id
                            name
                            price
                        }
                        quantity
                        status
                    }
                }
            }
        }
        """
        
        try:
            response = self._execute_query(query, {"userId": user_id})
            return response["data"]["user"]["supermarketLists"]
        except Exception as e:
            console.print(f"[bold red]Error fetching shopping history: {str(e)}[/bold red]")
            return []
