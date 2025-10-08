from shop.pagination import paginate


def test_pages_do_not_overlap():
    items = list(range(25))
    assert paginate(items, 1, 10)[-1] != paginate(items, 2, 10)[0]
