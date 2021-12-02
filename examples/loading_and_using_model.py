import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf
from neuralgpu import trainer, generators

DIR = './log/first-simple-trial'

sess = tf.Session()
model = trainer.load_model(sess, DIR)

example = generators.generators['baddet'].get_batch(8, 32)

result = model.step(sess, example, False)
print(result.to_string())
