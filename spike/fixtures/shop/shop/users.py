class User:
    def __init__(self, email):
        self.email = email.strip().lower()

    def domain(self):
        return self.email.split("@", 1)[1]
