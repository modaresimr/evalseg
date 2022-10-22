class CircleList(list):
    def __getitem__(self, item):
        if type(item) == int:
            return super().__getitem__(item % len(self))
        return super().__getitem__(item)
