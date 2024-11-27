class TestClassFactory:
    # Dictionary to store instances based on their initialization arguments
    _instances = {}
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        # Create a hashable key from the arguments
        # Convert args to a tuple (it already is one) and kwargs to a frozen set of items
        key = (args, frozenset(kwargs.items()))
        
        # If an instance with these arguments doesn't exist, create it
        if key not in cls._instances:
            cls._instances[key] = TestClass(*args, **kwargs)
            
        return cls._instances[key]
    
    @classmethod
    def clear_instances(cls):
        """Clear all stored instances"""
        cls._instances.clear()

class TestClass:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def __str__(self):
        return f"TestClass instance with args: {self.args}, kwargs: {self.kwargs}"

# Example usage
if __name__ == "__main__":
    # Getting instances with different arguments
    instance1 = TestClassFactory.get_instance(1, 2, name="first")
    instance2 = TestClassFactory.get_instance(1, 2, name="first")  # Same arguments
    instance3 = TestClassFactory.get_instance(3, 4, name="second")  # Different arguments
    
    # Verify instances are the same when arguments match
    print(f"instance1 is instance2: {instance1 is instance2}")  # True
    print(f"instance1 is instance3: {instance1 is instance3}")  # False
    
    # Print the instances
    print(f"Instance 1: {instance1}")
    print(f"Instance 2: {instance2}")
    print(f"Instance 3: {instance3}")