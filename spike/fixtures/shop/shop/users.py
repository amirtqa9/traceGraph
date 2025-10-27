class User:
    def __init__(self, email):
        self.raw_email = email.strip()
        self.email = self.raw_email.lower()

    def domain(self):
        return self.email.split("@", 1)[1]
