from shop.pagination import paginate


def search(catalog, query, page=1, size=10):
    query = query.lower()
    hits = [item for item in catalog.all() if query in item["name"].lower()]
    return paginate(hits, page, size)
