from shop.pricing import total


class Cart:
    def __init__(self):
        self.lines = {}

    def add(self, slug, price, qty=1):
        line = self.lines.setdefault(slug, {"price": price, "qty": 0})
        line["qty"] += qty

    def remove(self, slug):
        self.lines.pop(slug, None)

    def subtotal(self):
        return total(line["price"] * line["qty"] for line in self.lines.values())
