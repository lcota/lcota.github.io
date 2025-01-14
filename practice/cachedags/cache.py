import functools
import deltalake as dl
from deltalake import DeltaTable, write_deltalake
import pandas as pd

def cacheable(table_uri, name=None):
    def decorator_cacheable(func):
        @functools.wraps(func)
        def wrapper_cacheable(*args, **kwargs):
            # Print the name of the wrapped function            
            print(f"Wrapping function: {func.__name__}")
            if name is None:
                print(f"Cache name: {name}")
                
            
            # Execute the function and get the result
            result = func(*args, **kwargs)
            
            # Convert result to a DataFrame if it's not already
            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(result)
            
            # Write the result to the DeltaTable
            write_deltalake(table_or_uri=table_uri, data=result, mode='overwrite')
            
            return result
        return wrapper_cacheable
    return decorator_cacheable


class Cache:
    """
    A class to cache the results of a function into a DeltaTable.
    Attributes:
        table_uri (str): The URI of the DeltaTable.
        table (DeltaTable): The DeltaTable object.
    Methods:
        __call__(func):
            Decorator to cache the result of the function into the DeltaTable.
        read():
            Reads the data from the DeltaTable and returns it as a pandas DataFrame.
        write(data):
            Writes the given data to the DeltaTable.
        update(data):
            Updates the DeltaTable with the given data.
        delete(data):
            Deletes data from the DeltaTable based on a condition.
        vacuum():
            Performs garbage collection on the DeltaTable.
        history():
            Returns the history of the DeltaTable.
    Example:
        >>> cache = Cache(table_uri="path/to/delta/table")
        >>> @cache
        >>> def my_function():
        >>>     return {"key": "value"}
        >>> result = my_function()
        Wrapping function: my_function
        >>> print(result)
        {'key': 'value'}
        >>> df = cache.read()
        >>> print(df)
           key
        0  value
    """
    def __init__(self, table_uri):
        self.table_uri = table_uri
        self.table = DeltaTable(table_uri=table_uri)
        
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper_cacheable(*args, **kwargs):
            # Print the name of the wrapped function            
            print(f"Wrapping function: {func.__name__}")
            
            # Execute the function and get the result
            result = func(*args, **kwargs)
            
            # Convert result to a DataFrame if it's not already
            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(result)
            
            # Write the result to the DeltaTable
            write_deltalake(table_or_uri=self.table_uri, data=result, mode='overwrite')
            
            return result
        return wrapper_cacheable

    def read(self):
        return self.table.to_pandas()

    def write(self, data):
        write_deltalake(table_or_uri=self.table_uri, data=data, mode='overwrite')

    def update(self, data):
        self.table.update(data, "key = value")

    def delete(self, data):
        self.table.delete("key = value")

    def vacuum(self):
        self.table.vacuum()

    def history(self):
        return self.table.history()

