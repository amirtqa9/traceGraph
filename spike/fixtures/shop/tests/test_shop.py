import datetime as dt

from shop.cart import Cart
from shop.coupons import Coupon, is_valid, redeem
from shop.inventory import Inventory
from shop.pagination import paginate
from shop.shipping import shipping_cost
from shop.users import User


def test_pages_do_not_overlap():
    items = list(range(25))
    assert paginate(items, 1, 10)[-1] != paginate(items, 2, 10)[0]


def test_page_zero_is_first_page():
    assert paginate(list(range(25)), 0, 10) == list(range(10))


def test_discount():
    assert redeem(Coupon("X", 20, dt.date(2030, 1, 1)), 100.0, dt.date(2025, 1, 1)) == 80.0


def test_coupon_valid_on_expiry_day():
    day = dt.date(2025, 5, 1)
    assert is_valid(Coupon("X", 10, day), day)


def test_cart_quantity():
    cart = Cart()
    cart.add("a", 10.0)
    cart.add("a", 10.0)
    assert cart.subtotal() == 20.0


def test_free_shipping_boundary():
    assert shipping_cost(50.0) == 0.0


def test_no_overselling():
    inv = Inventory()
    inv.restock("a", 5)
    inv.reserve("a", 5)
    try:
        inv.reserve("a", 1)
    except ValueError:
        return
    raise AssertionError("oversold")


def test_domain_is_lowercase():
    assert User("Bob@Example.COM").domain() == "example.com"
