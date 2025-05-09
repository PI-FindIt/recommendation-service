import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table

from src.data.data_service import DataService
from src.data.test_data_generator import TestDataGenerator
from src.models.user_recommendation_engine import UserRecommendationEngine

console = Console()


class MockDataService(DataService):
    def __init__(self):
        super().__init__()
        self.test_generator = TestDataGenerator()
        self.users = self.test_generator.generate_users(10)

    def get_user_data(self, user_id: str) -> dict:
        """Override to return mock user data"""
        for user in self.users:
            if user["_id"] == user_id:
                return user
        raise ValueError(f"User {user_id} not found")


def test_recommendation_engine():
    console.print("\n[bold purple]=" * 50)
    console.print("[bold purple]Testing User Recommendation Engine")
    console.print("[bold purple]=" * 50)

    # Initialize engine with mock data service
    # Initialize data generator with real products
    data_generator = TestDataGenerator("products.json")

    # Print sample data to verify
    data_generator.print_sample_data()

    # Initialize engine with mock data service
    engine = UserRecommendationEngine()
    mock_data_service = MockDataService()
    mock_data_service.test_generator = data_generator
    engine.data_service = mock_data_service

    # Test recommendations for each user
    for user_id in track(range(10), description="Testing users..."):
        console.print(
            f"\n[bold cyan]Testing recommendations for User {user_id}[/bold cyan]"
        )

        # Get user data
        user_data = engine.data_service.get_user_data(str(user_id))

        # Print user preferences
        console.print("\n[yellow]User Preferences:[/yellow]")
        prefs_table = Table(show_header=True, header_style="bold magenta")
        prefs_table.add_column("Preference")
        prefs_table.add_column("Value")

        for key, value in user_data["preferences"].items():
            prefs_table.add_row(key, str(value))
        console.print(prefs_table)

        # Get recommendations
        try:
            recommendations = engine.get_recommendations(str(user_id), k=30)

            # Print recommendations
            console.print("\n[bold green]Top 5 Recommendations:[/bold green]")
            rec_table = Table(show_header=True, header_style="bold magenta")
            rec_table.add_column("Product")
            rec_table.add_column("Brand")
            rec_table.add_column("Category")
            rec_table.add_column("Score")
            rec_table.add_column("Factors")

            for rec in recommendations:
                product = rec["product"]
                factors = rec["scoring_factors"]
                factors_str = ", ".join(f"{k}: {v:.2f}" for k, v in factors.items())

                rec_table.add_row(
                    product["name"],
                    product.get("brandName", "N/A"),
                    product.get("categoryName", "N/A"),
                    f"{rec['score']:.3f}",
                    factors_str,
                )

            console.print(rec_table)

            # Print some analytics
            console.print("\n[cyan]Recommendation Analytics:[/cyan]")

            # Brand distribution
            brand_counts = {}
            for rec in recommendations:
                brand = rec["product"].get("brandName", "N/A")
                brand_counts[brand] = brand_counts.get(brand, 0) + 1

            console.print("\nBrand Distribution:")
            for brand, count in brand_counts.items():
                console.print(f"- {brand}: {count}")

            # Average score
            avg_score = np.mean([rec["score"] for rec in recommendations])
            console.print(f"\nAverage recommendation score: {avg_score:.3f}")

        except Exception as e:
            console.print(
                f"[bold red]Error getting recommendations: {str(e)}[/bold red]"
            )

        console.print("\n[bold purple]" + "=" * 50)


if __name__ == "__main__":
    test_recommendation_engine()
