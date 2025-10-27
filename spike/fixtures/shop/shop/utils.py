import re


def clamp(value, low, high):
    return max(low, min(value, high))


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def normalize(text):
    return " ".join(text.lower().split())
