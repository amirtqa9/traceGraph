def apply_discount(price, percent):
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * (100 - percent) / 100, 2)


def total(prices):
    return round(sum(prices), 2)
