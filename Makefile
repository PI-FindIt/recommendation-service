migration-new:
	docker compose -f ../compose.yaml exec product-service alembic revision --autogenerate -m "$(message)"

migration-upgrade:
	docker compose -f ../compose.yaml exec product-service alembic upgrade head

migration-downgrade:
	docker compose -f ../compose.yaml exec product-service alembic downgrade -1

merge-upstream-config:
	git checkout main
	git remote add upstream git@github.com:PI-FindIt/service-template.git
	git fetch upstream
	git merge upstream/main --allow-unrelated-histories

merge-upstream:
	git fetch upstream
	git merge upstream/main
