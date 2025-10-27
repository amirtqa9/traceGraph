FREE_SHIPPING_MIN = 50.0
FLAT_RATE = 4.99


def free_shipping_threshold():
    return FREE_SHIPPING_MIN


def shipping_cost(subtotal):
    return 0.0 if subtotal >= free_shipping_threshold() else FLAT_RATE
