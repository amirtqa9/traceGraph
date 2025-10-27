"""Fixture tasks for the M0 eval.

Each task plants one bug in the bug-free `fixtures/shop` repo and describes only its
symptom. `moved` is an alternative patch, also applied to the clean base, that produces
the same symptom from a *different* function. It is used by the `bug_moved` scenario,
where the world changes between the source session and the resume.

`check` is Python that sets `bug = True` when the symptom shows. Tests run it against
the base (must be False), the planted repo and the moved repo (both must be True), so a
fixture can't silently stop reproducing its symptom.

Patches are exact string replacements and must match the base exactly once.
Patched code must not contain comments that point at the bug.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "shop"

Patch = tuple[tuple[str, str, str], ...]  # (path, old, new)


@dataclass(frozen=True)
class FixtureTask:
    name: str
    symptom: str
    planted: Patch
    expected: tuple[str, str]  # (path, function) for the planted bug
    moved: Patch
    moved_expected: tuple[str, str]
    check: str

    @property
    def prompt(self) -> str:
        return f"{self.symptom} Find the function responsible."


def apply_patch(root: Path, patch: Patch) -> None:
    for rel, old, new in patch:
        path = root / rel
        text = path.read_text()
        if text.count(old) != 1:
            raise ValueError(f"patch for {rel} matches {text.count(old)} times: {old!r}")
        path.write_text(text.replace(old, new))


def materialize(dest: Path, patch: Patch = ()) -> Path:
    """Copy the clean fixture to `dest` and apply `patch`."""
    shutil.copytree(FIXTURE, dest, ignore=shutil.ignore_patterns("__pycache__"))
    apply_patch(dest, patch)
    return dest


TASKS: dict[str, FixtureTask] = {
    t.name: t
    for t in [
        FixtureTask(
            name="pagination_overlap",
            symptom="Users report that page 2 of search results starts with the same item that ended page 1.",
            planted=(("shop/pagination.py", "    end = start + size\n", "    end = start + size + 1\n"),),
            expected=("shop/pagination.py", "paginate"),
            moved=((
                "shop/pagination.py",
                "def paginate(items, page, size):\n"
                '    """Return the items on 1-based `page`."""\n'
                "    page = clamp(page, 1, page_count(len(items), size))\n"
                "    start = (page - 1) * size\n"
                "    end = start + size\n",
                "def page_bounds(page, size):\n"
                "    start = (page - 1) * size\n"
                "    return max(0, start - 1), start + size\n"
                "\n\n"
                "def paginate(items, page, size):\n"
                '    """Return the items on 1-based `page`."""\n'
                "    page = clamp(page, 1, page_count(len(items), size))\n"
                "    start, end = page_bounds(page, size)\n",
            ),),
            moved_expected=("shop/pagination.py", "page_bounds"),
            check=(
                "from shop.pagination import paginate\n"
                "items = list(range(25))\n"
                "bug = paginate(items, 1, 10)[-1] == paginate(items, 2, 10)[0]"
            ),
        ),
        FixtureTask(
            name="discount_inverted",
            symptom="A 20% discount coupon makes a $100 item cost $20 instead of $80.",
            planted=((
                "shop/pricing.py",
                "    return round(price * (100 - percent) / 100, 2)\n",
                "    return round(price * percent / 100, 2)\n",
            ),),
            expected=("shop/pricing.py", "apply_discount"),
            moved=(("shop/coupons.py", "    return coupon.value\n", "    return 100 - coupon.value\n"),),
            moved_expected=("shop/coupons.py", "coupon_percent"),
            check=(
                "import datetime as dt\n"
                "from shop.coupons import Coupon, redeem\n"
                "bug = redeem(Coupon('X', 20, dt.date(2030, 1, 1)), 100.0, dt.date(2025, 1, 1)) == 20.0"
            ),
        ),
        FixtureTask(
            name="cart_quantity",
            symptom="Adding the same item to the cart twice bills it only once.",
            planted=(("shop/cart.py", '        line["qty"] += qty\n', '        line["qty"] = qty\n'),),
            expected=("shop/cart.py", "add"),
            moved=((
                "shop/cart.py",
                '        return total(line["price"] * line["qty"] for line in self.lines.values())\n',
                '        return total(line["price"] for line in self.lines.values())\n',
            ),),
            moved_expected=("shop/cart.py", "subtotal"),
            check=(
                "from shop.cart import Cart\n"
                "c = Cart()\nc.add('a', 10.0)\nc.add('a', 10.0)\n"
                "bug = c.subtotal() == 10.0"
            ),
        ),
        FixtureTask(
            name="domain_case",
            symptom="User('Bob@Example.COM').domain() returns 'Example.COM' instead of 'example.com'.",
            planted=(("shop/users.py", "        self.email = self.raw_email.lower()\n",
                      "        self.email = self.raw_email\n"),),
            expected=("shop/users.py", "__init__"),
            moved=(("shop/users.py", '        return self.email.split("@", 1)[1]\n',
                    '        return self.raw_email.split("@", 1)[1]\n'),),
            moved_expected=("shop/users.py", "domain"),
            check="from shop.users import User\nbug = User('Bob@Example.COM').domain() != 'example.com'",
        ),
        FixtureTask(
            name="search_case",
            symptom="Searching for 'shirt' does not find the item named 'Blue SHIRT'.",
            planted=(("shop/search.py", '    return normalize(query) in normalize(item["name"])\n',
                      '    return normalize(query) in item["name"]\n'),),
            expected=("shop/search.py", "matches"),
            moved=(("shop/utils.py", '    return " ".join(text.lower().split())\n',
                    '    return " ".join(text.split())\n'),),
            moved_expected=("shop/utils.py", "normalize"),
            check=(
                "from shop.catalog import Catalog\nfrom shop.search import search\n"
                "c = Catalog()\nc.add('Blue SHIRT', 10)\n"
                "bug = search(c, 'shirt') == []"
            ),
        ),
        FixtureTask(
            name="free_shipping_boundary",
            symptom="An order with a subtotal of exactly $50.00 is charged shipping, but free shipping starts at $50.",
            planted=(("shop/shipping.py", "subtotal >= free_shipping_threshold()",
                      "subtotal > free_shipping_threshold()"),),
            expected=("shop/shipping.py", "shipping_cost"),
            moved=(("shop/shipping.py", "    return FREE_SHIPPING_MIN\n", "    return FREE_SHIPPING_MIN + 0.01\n"),),
            moved_expected=("shop/shipping.py", "free_shipping_threshold"),
            check="from shop.shipping import shipping_cost\nbug = shipping_cost(50.0) != 0.0",
        ),
        FixtureTask(
            name="overselling",
            symptom="Customers can reserve more units of an item than are in stock.",
            planted=(("shop/inventory.py", "        if qty > self.available(slug):\n",
                      "        if qty > self.stock.get(slug, 0):\n"),),
            expected=("shop/inventory.py", "reserve"),
            moved=(("shop/inventory.py",
                    "        return self.stock.get(slug, 0) - self.reserved.get(slug, 0)\n",
                    "        return self.stock.get(slug, 0) + self.reserved.get(slug, 0)\n"),),
            moved_expected=("shop/inventory.py", "available"),
            check=(
                "from shop.inventory import Inventory\n"
                "inv = Inventory()\ninv.restock('a', 5)\ninv.reserve('a', 5)\n"
                "try:\n    inv.reserve('a', 1)\n    bug = True\nexcept ValueError:\n    bug = False"
            ),
        ),
        FixtureTask(
            name="page_count_extra",
            symptom="With exactly 20 matching items and 10 per page, search reports 3 pages instead of 2.",
            planted=(("shop/pagination.py", "    return max(1, -(-total // size))\n",
                      "    return max(1, total // size + 1)\n"),),
            expected=("shop/pagination.py", "page_count"),
            moved=(("shop/search.py", "    return page_count(len(hits), size)\n",
                    "    return page_count(len(hits) + 1, size)\n"),),
            moved_expected=("shop/search.py", "total_pages"),
            check=(
                "from shop.catalog import Catalog\nfrom shop.search import total_pages\n"
                "c = Catalog()\n"
                "for i in range(20):\n    c.add(f'Shirt {i}', 1)\n"
                "bug = total_pages(c, 'shirt') != 2"
            ),
        ),
        FixtureTask(
            name="page_zero",
            symptom="Requesting page 0 of search results returns an empty list instead of the first page.",
            planted=(("shop/utils.py", "    return max(low, min(value, high))\n", "    return min(value, high)\n"),),
            expected=("shop/utils.py", "clamp"),
            moved=(("shop/pagination.py", "    page = clamp(page, 1, page_count(len(items), size))\n",
                    "    page = clamp(page, 0, page_count(len(items), size))\n"),),
            moved_expected=("shop/pagination.py", "paginate"),
            check="from shop.pagination import paginate\nbug = paginate(list(range(25)), 0, 10) == []",
        ),
        FixtureTask(
            name="coupon_expiry",
            symptom="Coupons are rejected on their expiry date; they should be valid through that day.",
            planted=(("shop/coupons.py", "    return days_left(coupon, today) >= 0\n",
                      "    return days_left(coupon, today) > 0\n"),),
            expected=("shop/coupons.py", "is_valid"),
            moved=(("shop/coupons.py", "    return (coupon.expires - today).days\n",
                    "    return (coupon.expires - today).days - 1\n"),),
            moved_expected=("shop/coupons.py", "days_left"),
            check=(
                "import datetime as dt\nfrom shop.coupons import Coupon, is_valid\n"
                "d = dt.date(2025, 5, 1)\n"
                "bug = not is_valid(Coupon('X', 10, d), d)"
            ),
        ),
    ]
}
