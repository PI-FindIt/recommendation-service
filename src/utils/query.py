PRODUCTS_QUERY = """
    query Products($filters: ProductFilter!) {
        products(filters: $filters) {
            ean
            name
            genericName
            brandName
            categoryName
            keywords
            ingredients
            nutriScore              
        }
    }
    """
