class Object:
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def __repr__(self):
        return "<" + self.name + "> " + str(self.__dict__)

    def __str__(self):
        return "<" + self.name + "> " + str(self.__dict__.keys())

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, data):
        return setattr(self, key, data)
