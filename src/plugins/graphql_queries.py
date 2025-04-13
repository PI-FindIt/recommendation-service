USERS_QUERY = """
query GetUsers($userId: String!) {
  user(id: $userId) {
    _id
    preferences {
      brandsLike
      supermarketsLike
      maxDistance
      pathType
    }
    supermarketLists {
      _id
      products {
        product {
          ean  # Extensão do serviço USERS
          
            name
            brandName
            categoryName
          
        }
        supermarket {
          id  # Extensão do serviço USERS
         
            name
            locations {
              latitude
              longitude
            }
         
        }
        quantity
        status
      }
    }
  }
}
"""

PRODUCTS_QUERY = """
query GetProducts($filters: ProductFilter!) {
  products(filters: $filters) {
    ean
    name
    nutriScore
    brandName
    categoryName
    keywords
    nutrition
    supermarkets {  # Do serviço SUPERMARKETS
      price
      supermarket {
        id
        name
      }
    }
  }
}
"""