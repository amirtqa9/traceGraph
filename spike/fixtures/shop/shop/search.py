from shop.pagination import page_count, paginate
from shop.utils import normalize


def matches(item, query):
    return normalize(query) in normalize(item["name"])


def search(catalog, query, page=1, size=10):
    hits = [item for item in catalog.all() if matches(item, query)]
    return paginate(hits, page, size)


def total_pages(catalog, query, size=10):
    hits = [item for item in catalog.all() if matches(item, query)]
    return page_count(len(hits), size)
