from shop.pricing import apply_discount


class Coupon:
    def __init__(self, code, value, expires):
        self.code = code
        self.value = value  # percent off
        self.expires = expires


def coupon_percent(coupon):
    return coupon.value


def days_left(coupon, today):
    return (coupon.expires - today).days


def is_valid(coupon, today):
    return days_left(coupon, today) >= 0


def redeem(coupon, price, today):
    if not is_valid(coupon, today):
        raise ValueError("coupon expired")
    return apply_discount(price, coupon_percent(coupon))
