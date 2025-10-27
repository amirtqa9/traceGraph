from shop.utils import clamp


def page_count(total, size):
    return max(1, -(-total // size))


def paginate(items, page, size):
    """Return the items on 1-based `page`."""
    page = clamp(page, 1, page_count(len(items), size))
    start = (page - 1) * size
    end = start + size
    return items[start:end]
