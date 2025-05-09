import numpy as np
from rich.console import Console

from src.data.data_service import DataService
from src.models.base_engine import BaseEngine
from src.models.product_similarity import ProductSimilarityEngine

console = Console()


class UserRecommendationEngine(BaseEngine):
    def __init__(self):
        field_weights = {
            "product_similarity": 0.3,
            "category_affinity": 0.2,
            "brand_affinity": 0.2,
            "price_affinity": 0.15,
            "shopping_history": 0.15,
        }
        super().__init__(field_weights=field_weights, engine_name="user_recommendation")

        self.product_similarity_engine = ProductSimilarityEngine(print=False)
        self.product_similarity_engine.load_embeddings(load_format="faiss")

        self.data_service = DataService()

    def _get_user_data(self, user_id: str) -> dict:
        """Get user data from GraphQL API"""
        return self.data_service.get_user_data(user_id)

    def _get_user_preferences(self, user_id: str) -> dict:
        # filter out the user data
        user_data = self._get_user_data(user_id)
        return user_data.get("preferences", {})

    def _get_products_by_brand(self, brand: str) -> list[dict]:
        """Get products by brand from GraphQL API"""
        return self.data_service.get_products_by_brand(brand)

    def _get_products_by_category(self, category: str) -> list[dict]:
        """Get products by category from GraphQL API"""
        return self.data_service.get_products_by_category(category)

    def _generate_field_embeddings(self, user_data: dict) -> dict[str, np.ndarray]:
        embeddings = {}

        # Brand affinity embeddings
        if user_data.get("preferences", {}).get("brandsLike"):
            brands_text = " ".join(user_data["preferences"]["brandsLike"])
            embeddings["brand_affinity"] = self.model.encode(
                brands_text, convert_to_numpy=True, device=self.device
            )
        if user_data.get("preferences", {}).get("brandsDislike"):
            brands_text = " ".join(user_data["preferences"]["brandsDislike"])
            if "brand_affinity" in embeddings:
                embeddings["brand_affinity"] -= self.model.encode(
                    brands_text, convert_to_numpy=True, device=self.device
                )
            else:
                embeddings["brand_affinity"] = -self.model.encode(
                    brands_text, convert_to_numpy=True, device=self.device
                )

        # Supermarket affinity embeddings
        if user_data.get("preferences", {}).get("supermarketsLike"):
            supermarkets_text = " ".join(
                map(str, user_data["preferences"]["supermarketsLike"])
            )
            embeddings["supermarket_affinity"] = self.model.encode(
                supermarkets_text, convert_to_numpy=True, device=self.device
            )
        if user_data.get("preferences", {}).get("supermarketsDislike"):
            supermarkets_text = " ".join(
                map(str, user_data["preferences"]["supermarketsDislike"])
            )
            if "supermarket_affinity" in embeddings:
                embeddings["supermarket_affinity"] -= self.model.encode(
                    supermarkets_text, convert_to_numpy=True, device=self.device
                )
            else:
                embeddings["supermarket_affinity"] = -self.model.encode(
                    supermarkets_text, convert_to_numpy=True, device=self.device
                )

        # Shopping history embeddings
        if user_data.get("supermarketLists"):
            history_embeddings = []
            for supermarket_list in user_data["supermarketLists"]:
                if supermarket_list.get("products"):
                    for product in supermarket_list["products"]:
                        if product.get("product", {}).get("ean"):
                            try:
                                # Find product index in similarity engine
                                product_idx = self._find_product_index(
                                    product["product"]["ean"]
                                )
                                if product_idx is not None:
                                    product_embedding = self.product_similarity_engine.index.reconstruct(
                                        product_idx
                                    )
                                    history_embeddings.append(product_embedding)
                            except Exception as e:
                                console.print(
                                    f"[yellow]Warning: Could not process product {product['product']['ean']}: {str(e)}[/yellow]"
                                )

            if history_embeddings:
                embeddings["shopping_history"] = np.mean(history_embeddings, axis=0)

        # Price sensitivity embedding (based on purchase history average prices)
        if user_data.get("supermarketLists"):
            prices = []
            for supermarket_list in user_data["supermarketLists"]:
                if supermarket_list.get("products"):
                    for product in supermarket_list["products"]:
                        if product.get("supermarket", {}).get("price"):
                            prices.append(float(product["supermarket"]["price"]))
            if prices:
                embeddings["price_affinity"] = np.array([np.mean(prices)])

        return embeddings

    def generate_candidates(self, user_id: str, k: int = 100) -> list[dict]:
        """
        Step 1: Candidate Generation
        Generate initial candidates through multiple strategies
        """
        candidates = []

        # Strategy 1: Products similar to previously purchased items
        history_candidates = self._get_history_based_candidates(user_id)

        # Strategy 2: Products from preferred brands
        brand_candidates = self._get_brand_based_candidates(user_id)

        # Strategy 3: Popular products in user's preferred categories
        category_candidates = self._get_category_based_candidates(user_id)

        # Combine and deduplicate candidates
        candidates = self._merge_candidates(
            history_candidates, brand_candidates, category_candidates
        )

        return candidates[:k]

    def score_candidates(
        self,
        user_id: str,
        candidates: list[dict],
        user_context: (
            dict | None
        ) = None,  # Pode ser util para real time things like the hour of the day
    ) -> list[dict]:
        """
        Step 2: Scoring
        Score each candidate based on multiple factors
        """
        scored_candidates = []

        for candidate in candidates:
            score = 0.0

            # Factor 1: Product similarity score
            sim_score = self._calculate_similarity_score(candidate, user_id)
            score += self.field_weights["product_similarity"] * sim_score

            # Factor 2: Brand affinity score
            brand_score = self._calculate_brand_affinity(candidate, user_id)
            score += self.field_weights["brand_affinity"] * brand_score

            # Factor 3: Price affinity score
            price_score = self._calculate_price_affinity(candidate, user_id)
            score += self.field_weights["price_affinity"] * price_score

            scored_candidates.append(
                {
                    "product": candidate,
                    "score": score,
                    "scoring_factors": {
                        "similarity": sim_score,
                        "brand_affinity": brand_score,
                        "price_affinity": price_score,
                    },
                }
            )

        return sorted(scored_candidates, key=lambda x: x["score"], reverse=True)

    def rerank_candidates(
        self, user_id: str, scored_candidates: list[dict], user_preferences: dict
    ) -> list[dict]:
        """
        Step 3: Re-ranking
        Apply business rules and user preferences to final ranking
        """
        reranked = scored_candidates.copy()

        # Apply budget constraints
        if user_preferences.get("budget", -1) > 0:
            reranked = self._filter_by_budget(reranked, user_preferences["budget"])

        # Boost products from preferred supermarkets
        if user_preferences.get("supermarketsLike"):
            reranked = self._boost_preferred_supermarkets(
                reranked, user_preferences["supermarketsLike"]
            )

        # Penalize or remove products from disliked brands
        if user_preferences.get("brandsDislike"):
            reranked = self._penalize_disliked_brands(
                reranked, user_preferences["brandsDislike"]
            )

        # Apply diversity rules (avoid too many similar products)
        reranked = self._apply_diversity_rules(reranked)

        return reranked

    def get_recommendations(
        self, user_id: str, k: int = 10, context: dict | None = None
    ) -> list[dict]:
        """
        Main recommendation pipeline
        """
        # Step 1: Generate candidates
        candidates = self.generate_candidates(user_id, k=min(k * 10, 10000))

        # Step 2: Score candidates
        scored_candidates = self.score_candidates(user_id, candidates, context)

        # Step 3: Re-rank based on business rules and user preferences
        user_preferences = self._get_user_preferences(user_id)
        final_recommendations = self.rerank_candidates(
            user_id, scored_candidates, user_preferences
        )

        console.print(f"Final recommendations: {final_recommendations}")

        return final_recommendations[:k]

    def _find_product_index(self, ean: str) -> int | None:
        """Helper function to find product index in similarity engine"""
        for idx, product in self.product_similarity_engine.data_mapping.items():
            if product["ean"] == ean:
                return idx
        return None

    def _get_history_based_candidates(self, user_id: str) -> list[dict]:
        """Get candidates based on user's shopping history"""
        candidates = []
        user_data = self._get_user_data(user_id)

        if user_data.get("supermarketLists"):
            for supermarket_list in user_data["supermarketLists"]:
                if supermarket_list.get("products"):
                    for product in supermarket_list["products"]:
                        if product.get("product", {}).get("ean"):
                            try:
                                similar_products = self.product_similarity_engine.get_product_recommendations(
                                    product["product"]["ean"], k=5
                                )
                                candidates.extend(
                                    [p["product"] for p in similar_products]
                                )
                            except Exception as e:
                                console.print(
                                    f"[yellow]Warning: Could not get similar products for {product['product']['ean']}: {str(e)}[/yellow]"
                                )

        return candidates

    def _get_brand_based_candidates(self, user_id: str) -> list[dict]:
        """Get candidates based on user's preferred brands"""
        candidates = []
        user_data = self._get_user_data(user_id)

        if user_data.get("preferences", {}).get("brandsLike"):
            for brand in user_data["preferences"]["brandsLike"]:
                brand_products = self._get_products_by_brand(brand)
                candidates.extend(brand_products)

        return candidates

    def _get_category_based_candidates(self, user_id: str) -> list[dict]:
        """Get candidates based on categories from user's history"""
        candidates = []
        user_data = self._get_user_data(user_id)

        categories = set()
        if user_data.get("supermarketLists"):
            for supermarket_list in user_data["supermarketLists"]:
                if supermarket_list.get("products"):
                    for product in supermarket_list["products"]:
                        if product.get("product", {}).get("categoryName"):
                            categories.add(product["product"]["categoryName"])

        for category in categories:
            category_products = self._get_products_by_category(category)
            candidates.extend(category_products)

        return candidates

    def _merge_candidates(self, *candidate_lists: list[dict]) -> list[dict]:
        """Merge and deduplicate candidates"""
        seen_eans = set()
        merged = []

        for candidates in candidate_lists:
            for product in candidates:
                if product["ean"] not in seen_eans:
                    merged.append(product)
                    seen_eans.add(product["ean"])

        return merged

    def _calculate_similarity_score(self, candidate: dict, user_id: str) -> float:
        """Calculate similarity score between candidate and user's history"""
        user_data = self._get_user_data(user_id)
        if not user_data.get("supermarketLists"):
            return 0.0

        try:
            candidate_idx = self._find_product_index(candidate["ean"])
            if candidate_idx is None:
                return 0.0

            candidate_embedding = self.product_similarity_engine.index.reconstruct(
                candidate_idx
            )

            # Get average similarity with historical products
            similarities = []
            for supermarket_list in user_data["supermarketLists"]:
                if supermarket_list.get("products"):
                    for product in supermarket_list["products"]:
                        if product.get("product", {}).get("ean"):
                            hist_idx = self._find_product_index(
                                product["product"]["ean"]
                            )
                            if hist_idx is not None:
                                hist_embedding = (
                                    self.product_similarity_engine.index.reconstruct(
                                        hist_idx
                                    )
                                )
                                similarity = 1.0 / (
                                    1.0
                                    + np.linalg.norm(
                                        candidate_embedding - hist_embedding
                                    )
                                )
                                similarities.append(similarity)

            return np.mean(similarities) if similarities else 0.0
        except Exception as e:
            console.print(
                f"[yellow]Warning: Error calculating similarity score: {str(e)}[/yellow]"
            )
            return 0.0

    def _calculate_brand_affinity(self, candidate: dict, user_id: str) -> float:
        """Calculate brand affinity score"""
        user_data = self._get_user_data(user_id)
        if not candidate.get("brandName"):
            return 0.0

        score = 0.0
        if user_data.get("preferences", {}).get("brandsLike"):
            if candidate["brandName"] in user_data["preferences"]["brandsLike"]:
                score += 1.0

        if user_data.get("preferences", {}).get("brandsDislike"):
            if candidate["brandName"] in user_data["preferences"]["brandsDislike"]:
                score -= 1.0

        return score

    def _calculate_price_affinity(self, candidate: dict, user_id: str) -> float:
        """Calculate price affinity score"""
        user_data = self._get_user_data(user_id)
        if not candidate.get("supermarkets"):
            return 0.0

        # Get average price of the candidate product
        prices = [s["price"] for s in candidate["supermarkets"]]
        if not prices:
            return 0.0

        avg_price = np.mean(prices)

        # Compare with user's typical spending
        user_avg_price = 0.0
        price_count = 0
        if user_data.get("supermarketLists"):
            for supermarket_list in user_data["supermarketLists"]:
                if supermarket_list.get("products"):
                    for product in supermarket_list["products"]:
                        if product.get("supermarket", {}).get("price"):
                            user_avg_price += float(product["supermarket"]["price"])
                            price_count += 1

        if price_count == 0:
            return 0.0

        user_avg_price /= price_count

        # Calculate score based on price difference
        price_diff = abs(avg_price - user_avg_price) / max(user_avg_price, 1.0)
        return 1.0 / (1.0 + price_diff)

    def _filter_by_budget(self, candidates: list[dict], budget: float) -> list[dict]:
        """Filter candidates by budget constraint"""
        if budget <= 0:
            return candidates

        filtered = []
        for candidate in candidates:
            if candidate.get("product", {}).get("supermarkets"):
                min_price = min(
                    s["price"] for s in candidate["product"]["supermarkets"]
                )
                if min_price <= budget:
                    filtered.append(candidate)

        return filtered

    def _boost_preferred_supermarkets(
        self, candidates: list[dict], preferred_supermarkets: list[int]
    ) -> list[dict]:
        """Boost scores of products available in preferred supermarkets"""
        boosted = []
        boost_factor = 1.2

        for candidate in candidates:
            score = candidate["score"]
            if a := candidate.get("product", {}).get("supermarkets"):
                print("FARTO ", a)
                supermarket_ids = [
                    s["supermarket"]["id"] for s in candidate["product"]["supermarkets"]
                ]
                if any(sid in preferred_supermarkets for sid in supermarket_ids):
                    score *= boost_factor

            boosted.append({**candidate, "score": score})

        return sorted(boosted, key=lambda x: x["score"], reverse=True)

    def _penalize_disliked_brands(
        self, candidates: list[dict], disliked_brands: list[str]
    ) -> list[dict]:
        """Penalize or remove products from disliked brands"""
        filtered = []
        penalty_factor = 0.5

        for candidate in candidates:
            if candidate.get("product", {}).get("brandName") in disliked_brands:
                candidate["score"] *= penalty_factor
            filtered.append(candidate)

        return sorted(filtered, key=lambda x: x["score"], reverse=True)

    def _apply_diversity_rules(self, candidates: list[dict]) -> list[dict]:
        """Apply diversity rules to avoid similar products in top recommendations"""
        if not candidates:
            return []

        diverse_candidates = [candidates[0]]
        similarity_threshold = 0.8

        for candidate in candidates[1:]:
            is_diverse = True
            candidate_idx = self._find_product_index(candidate["product"]["ean"])

            if candidate_idx is not None:
                candidate_embedding = self.product_similarity_engine.index.reconstruct(
                    candidate_idx
                )

                for selected in diverse_candidates:
                    selected_idx = self._find_product_index(selected["product"]["ean"])
                    if selected_idx is not None:
                        selected_embedding = (
                            self.product_similarity_engine.index.reconstruct(
                                selected_idx
                            )
                        )
                        similarity = 1.0 / (
                            1.0
                            + np.linalg.norm(candidate_embedding - selected_embedding)
                        )

                        if similarity > similarity_threshold:
                            is_diverse = False
                            break

            if is_diverse:
                diverse_candidates.append(candidate)

        return diverse_candidates
