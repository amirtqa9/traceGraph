from shop.utils import slugify


class Catalog:
    def __init__(self):
        self._items = []

    def add(self, name, price):
        self._items.append({"name": name, "slug": slugify(name), "price": price})

    def all(self):
        return sorted(self._items, key=lambda item: item["slug"])
