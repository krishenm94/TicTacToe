class Transform:
    """docstring for Transform"""

    def __init__(self, operations):
        self.operations = operations

    def execute(self, array):
        for operation in self.operations:
            array = operation(array)

        return array

