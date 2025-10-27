from shop.shipping import shipping_cost


def order_total(cart):
    subtotal = cart.subtotal()
    return round(subtotal + shipping_cost(subtotal), 2)


def place(cart, inventory):
    for slug, line in cart.lines.items():
        inventory.reserve(slug, line["qty"])
    return order_total(cart)
