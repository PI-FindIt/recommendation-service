PRODUCTS_QUERY = """
    query Products($filters: ProductFilter!) {
        products(filters: $filters) {
            ean
            name
            nameEn
            genericName
            genericNameEn
            brandName
            categoryName
            keywords
            ingredients
            nutriScore              
        }
    }
    """
