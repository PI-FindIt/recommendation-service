import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

from rich.console import Console

console = Console()


class TestDataGenerator:
    def __init__(self, products_file: str = "products2.json"):
        # Load real products data
        try:
            with open(products_file, "r") as f:
                self.products = json.load(f)
                self.products = self.products["data"]["products"]
            console.print(
                f"[green]Loaded {len(self.products)} products from {products_file}[/green]"
            )
        except Exception as e:
            console.print(f"[bold red]Error loading products: {str(e)}[/bold red]")
            self.products = []

        # Extract unique brands, categories, and create product lookup
        self.brands = list(
            set(p.get("brandName") for p in self.products if p.get("brandName"))
        )
        self.categories = list(
            set(p.get("categoryName") for p in self.products if p.get("categoryName"))
        )
        self.supermarket_ids = list(range(1, 7))  # Assuming 6 supermarkets

        # Create EAN lookup for quick access
        self.ean_lookup = {}
        for product in self.products:
            if product.get("ean"):
                self.ean_lookup[product["ean"]] = product

        console.print(f"[blue]Found {len(self.brands)} unique brands[/blue]")
        console.print(f"[blue]Found {len(self.categories)} unique categories[/blue]")

    def _get_random_product(self) -> Dict[str, Any]:
        """Get a random product from the real products data"""
        return random.choice(self.products)

    def generate_user(self, user_id: str) -> Dict[str, Any]:
        """Generate a mock user with preferences and shopping history using real product data"""

        # Generate random preferences from real brands
        liked_brands = random.sample(
            self.brands, min(random.randint(1, 3), len(self.brands))
        )
        remaining_brands = [b for b in self.brands if b not in liked_brands]
        disliked_brands = random.sample(
            remaining_brands, min(random.randint(0, 2), len(remaining_brands))
        )

        liked_supermarkets = random.sample(self.supermarket_ids, random.randint(1, 3))
        remaining_supermarkets = [
            s for s in self.supermarket_ids if s not in liked_supermarkets
        ]
        disliked_supermarkets = random.sample(
            remaining_supermarkets, random.randint(0, 2)
        )

        preferences = {
            "brandsLike": liked_brands,
            "brandsDislike": disliked_brands,
            "supermarketsLike": liked_supermarkets,
            "supermarketsDislike": disliked_supermarkets,
            "maxDistance": random.uniform(5.0, 20.0),
            "budget": random.uniform(50.0, 200.0),
            "pathType": random.choice(["SHORTEST", "CHEAPEST", "FASTEST"]),
        }

        # Generate shopping history with real products
        num_lists = random.randint(2, 5)
        shopping_lists = []

        for _ in range(num_lists):
            products = []
            num_products = random.randint(3, 10)

            # Select random products from real data
            for _ in range(num_products):
                real_product = self._get_random_product()

                product = {
                    "product": {
                        "ean": real_product["ean"],
                        "name": real_product["name"],
                        "brandName": real_product.get("brandName"),
                        "categoryName": real_product.get("categoryName"),
                    },
                    "supermarket": {
                        "id": random.choice(self.supermarket_ids),
                        "name": f"Supermarket_{random.choice(self.supermarket_ids)}",
                        "price": random.uniform(0.5, 50.0),
                    },
                    "quantity": random.randint(1, 5),
                    "status": random.choice(["COMPLETED", "SKIPPED", "ACTIVE"]),
                }
                products.append(product)

            shopping_list = {
                "_id": f"list_{random.randint(1000, 9999)}",
                "timestamp": (
                    datetime.now() - timedelta(days=random.randint(0, 30))
                ).isoformat(),
                "status": "COMPLETED",
                "products": products,
            }
            shopping_lists.append(shopping_list)

        return {
            "_id": user_id,
            "first_name": f"User_{user_id}",
            "last_name": "Test",
            "preferences": preferences,
            "supermarketLists": shopping_lists,
        }

    def generate_users(self, num_users: int) -> List[Dict[str, Any]]:
        """Generate multiple test users"""
        return [self.generate_user(str(i)) for i in range(num_users)]

    def print_sample_data(self):
        """Print sample of generated data for verification"""
        console.print("\n[bold cyan]Sample Generated Data[/bold cyan]")

        # Print some brands and categories
        console.print("\n[yellow]Sample Brands:[/yellow]")
        for brand in self.brands[:5]:
            console.print(f"- {brand}")

        console.print("\n[yellow]Sample Categories:[/yellow]")
        for category in self.categories[:5]:
            console.print(f"- {category}")

        # Generate and print a sample user
        sample_user = self.generate_user("sample")
        console.print("\n[yellow]Sample User Preferences:[/yellow]")
        for key, value in sample_user["preferences"].items():
            console.print(f"- {key}: {value}")

        console.print("\n[yellow]Sample Shopping History:[/yellow]")
        for product in sample_user["supermarketLists"][0]["products"][:3]:
            console.print(
                f"- {product['product']['name']} (EAN: {product['product']['ean']})"
            )
