# You may import any python packages that will be available in the environment you will run this algorithm in
# Environments can change based on the experiment's settings
import numpy

class Algorithm:
    # initialise fields to store cross-input data (e.g. machines, aggregations, etc.)
    def setup(self, parameters):
        # Retrieve the value of the parameters
        self.language = parameters['language']
        return True

    # this will be called each time the sync'd input has more data available to be processed
    def process(self, inputs, data_loaders, outputs):

        print("mt_preprocessing: LANG = ", self.language)
        raw_text = inputs['raw_text'].data.text
        print("{}\n".format(raw_text))

        outputs['tokenized_text'].write({'text':raw_text})

        # always return True, it signals BEAT to continue processing
        return True







