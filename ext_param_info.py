import logging

import numpy

import cPickle

from blocks.extensions import SimpleExtension

logging.basicConfig(level='INFO')
logger = logging.getLogger('extensions.ParamInfo')

class ParamInfo(SimpleExtension):
	def __init__(self, model, **kwargs):
		super(ParamInfo, self).__init__(**kwargs)

		self.model = model
	
	def do(self, which_callback, *args):
		print("---- PARAMETER INFO ----")
		print("\tmin\tmax\tmean\tvar\tdim\t\tname")
		for k, v in self.model.get_parameter_values().iteritems():
			print("\t%.4f\t%.4f\t%.4f\t%.4f\t%13s\t%s"%
					(v.min(), v.max(), v.mean(), ((v-v.mean())**2).mean(), 'x'.join([repr(x) for x in v.shape]), k))
		
