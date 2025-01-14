import logging
import sys
import hamilton

import pandas as pd

# We add this to speed up running things if you have a lot in your python environment.
from hamilton import registry; registry.disable_autoload()
from hamilton import driver, base
import features  # we import the module here!


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout)

if __name__ == '__main__':
    # Instantiate a common spine for your pipeline
    index = pd.date_range("2022-01-01", periods=6, freq="w")
    initial_columns = {  # load from actuals or wherever -- this is our initial data we use as input.
        # Note: these do not have to be all series, they could be scalar inputs.
        'signups': pd.Series([1, 10, 50, 100, 200, 400], index=index),
        'spend': pd.Series([10, 10, 20, 40, 40, 50], index=index),
    }
    dr = (
      driver.Builder()
        .with_config({})  # we don't have any configuration or invariant data for this example.
        .with_modules(features)  # we need to tell hamilton where to load function definitions from
        .with_adapters(base.PandasDataFrameResult())  # we want a pandas dataframe as output
        .build()
    )
    # we need to specify what we want in the final dataframe (these could be function pointers).
    output_columns = [
        'spend',
        'signups',
        'avg_3wk_spend',
        'acquisition_cost',
    ]
    # let's create the dataframe!
    df = dr.execute(output_columns, inputs=initial_columns)
    print(df)
    # `pip install sf-hamilton[visualization]` earlier you can also do
    dr.visualize_execution(bypass_validation=True,
                           final_vars=output_columns, 
                           output_file_path='./my_dag.png')
    

