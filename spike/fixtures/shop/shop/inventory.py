class Inventory:
    def __init__(self):
        self.stock = {}
        self.reserved = {}

    def restock(self, slug, qty):
        self.stock[slug] = self.stock.get(slug, 0) + qty

    def available(self, slug):
        return self.stock.get(slug, 0) - self.reserved.get(slug, 0)

    def reserve(self, slug, qty):
        if qty > self.available(slug):
            raise ValueError("insufficient stock")
        self.reserved[slug] = self.reserved.get(slug, 0) + qty
