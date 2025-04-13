from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.redis.hooks.redis import RedisHook
from datetime import datetime, timedelta
from src.plugins import graphql_queries, helpers
import httpx
import json
import logging
import numpy as np

default_args = {
    'owner': 'airflow',
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

def fetch_user_ids():
    """Obter todos os IDs de usuário do MongoDB"""
    # Implementação fictícia - substituir por conexão real
    return ["user1", "user2", "user3"]

def extract_users():
    users = []
    user_ids = fetch_user_ids()

    for user_id in user_ids:
        try:
            response = httpx.post(
                "http://findit-user-service-1:8000/graphql",
                json={
                    "query": graphql_queries.USERS_QUERY,
                    "variables": {"userId": user_id}
                },
                timeout=30
            ).json()

            if 'errors' in response:
                logging.error(f"Error fetching user {user_id}: {response['errors']}")
                continue

            users.append(response['data']['user'])

        except Exception as e:
            logging.error(f"Failed to fetch user {user_id}: {str(e)}")

    return users

def extract_products():
    try:
        response = httpx.post(
            "http://findit-product-service-1:8000/graphql",
            json={"query": graphql_queries.PRODUCTS_QUERY},
            timeout=30
        ).json()

        if 'errors' in response:
            logging.error(f"Product fetch error: {response['errors']}")
            return []

        return response['data']['products']

    except Exception as e:
        logging.error(f"Product extraction failed: {str(e)}")
        return []

def transform_users(ti):
    raw_users = ti.xcom_pull(task_ids='extract_users')
    transformed = []

    for user in raw_users:
        try:
            transformed.append({
                'user_id': user['_id'],
                'preferences': user['preferences'],
                'purchase_history': [
                    {
                        'ean': p['product']['ean'],
                        'product_name': p['product'].get('name', ''),
                        'brand': p['product'].get('brandName', ''),
                        'nutri_score': p['product'].get('nutriScore', 'UNKNOWN'),
                        'supermarket_id': p['supermarket']['id'],
                        'supermarket_name': p['supermarket'].get('name', ''),
                        'quantity': p['quantity'],
                        'price': None  # Preenchido na transformação de produtos
                    }
                    for lst in user['supermarketLists']
                    for p in lst['products']
                ]
            })
        except KeyError as e:
            logging.error(f"Invalid user data structure: {str(e)}")

    return transformed

def transform_products(ti):
    raw_products = ti.xcom_pull(task_ids='extract_products')
    transformed = []

    for product in raw_products:
        try:
            prices = [s['price'] for s in product['supermarkets']]
            transformed.append({
                'ean': product['ean'],
                'name': product['name'],
                'nutri_score': product['nutriScore'],
                'brand': product['brandName'],
                'keywords': product['keywords'],
                'nutrition': helpers.transform_nutrition(product['nutrition']),
                'price_stats': helpers.calculate_price_stats(prices),
                'embedding': helpers.generate_product_embeddings(product['keywords'])
            })
        except KeyError as e:
            logging.error(f"Invalid product data: {str(e)}")

    return transformed

def load_to_postgres(ti, table_name):
    postgres_hook = PostgresHook(postgres_conn_id='recommendations_db')
    conn = postgres_hook.get_conn()
    cursor = conn.cursor()

    data = ti.xcom_pull(task_ids=f'transform_{table_name}')

    if table_name == 'users':
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                preferences JSONB,
                purchase_history JSONB
            )
        ''')

        for record in data:
            cursor.execute('''
                INSERT INTO users (user_id, preferences, purchase_history)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                preferences = EXCLUDED.preferences,
                purchase_history = EXCLUDED.purchase_history
            ''', (
                record['user_id'],
                json.dumps(record['preferences']),
                json.dumps(record['purchase_history'])
            ))

    elif table_name == 'products':
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                ean TEXT PRIMARY KEY,
                name TEXT,
                nutri_score TEXT,
                brand TEXT,
                keywords JSONB,
                nutrition JSONB,
                price_stats JSONB,
                embedding VECTOR(384)
        ''')

        for record in data:
            cursor.execute('''
                INSERT INTO products 
                (ean, name, nutri_score, brand, keywords, nutrition, price_stats, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ean) DO UPDATE SET
                name = EXCLUDED.name,
                nutri_score = EXCLUDED.nutri_score,
                brand = EXCLUDED.brand,
                keywords = EXCLUDED.keywords,
                nutrition = EXCLUDED.nutrition,
                price_stats = EXCLUDED.price_stats,
                embedding = EXCLUDED.embedding
            ''', (
                record['ean'],
                record['name'],
                record['nutri_score'],
                record['brand'],
                json.dumps(record['keywords']),
                json.dumps(record['nutrition']),
                json.dumps(record['price_stats']),
                np.array(record['embedding'])
            ))

    conn.commit()
    cursor.close()

with DAG(
        'recommendations_etl',
        default_args=default_args,
        start_date=datetime(2024, 1, 1),
        schedule_interval='@daily',
        catchup=False
) as dag:

    extract_users_task = PythonOperator(
        task_id='extract_users',
        python_callable=extract_users
    )

    extract_products_task = PythonOperator(
        task_id='extract_products',
        python_callable=extract_products
    )

    transform_users_task = PythonOperator(
        task_id='transform_users',
        python_callable=transform_users
    )

    transform_products_task = PythonOperator(
        task_id='transform_products',
        python_callable=transform_products
    )

    load_users_task = PythonOperator(
        task_id='load_users',
        python_callable=load_to_postgres,
        op_kwargs={'table_name': 'users'}
    )

    load_products_task = PythonOperator(
        task_id='load_products',
        python_callable=load_to_postgres,
        op_kwargs={'table_name': 'products'}
    )

    extract_users_task >> transform_users_task >> load_users_task
    extract_products_task >> transform_products_task >> load_products_task