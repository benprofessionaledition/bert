"""
Runs the BERT classification task for the K12 usecase. The input file should just be a newline delimited list of comments.
Usage:
    python predict_k12.py <input_file> [output_file]
"""

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections

"""BERT finetuning runner."""

import tensorflow as tf
from tensorflow.python import debug as tfdbg
import tokenization
import modeling
import os
import sys
import numpy as np
import pandas as pd
from run_classifier import model_fn_builder, ManaProcessor169, file_based_convert_examples_to_features, file_based_input_fn_builder

# Required parameters that i'm hard coding


BERT_CONFIG_FILE = '.models/uncased_L-24_H-1024_A-16/bert_config.json'
TASK_NAME = 'mana169'
VOCAB_FILE = '.models/uncased_L-24_H-1024_A-16/vocab.txt'
# initial checkpoint
INIT_CKPT = '.models/fine_tuned/model/model.ckpt-5658'
# whether to use a lower case-only model (yes)
DO_LOWER_CASE = True
MAX_SEQ_LENGTH = 64
DO_PREDICT = True
PREDICT_BATCH_SIZE = 32
OUTPUT_DIR = 'output'

DONT_CARE = 1

input_df = None

def read_input_examples(filename):
    """
    BERT expects a TFRecordDataset, and the only way I've ever seen this
    done is by serializing and then deserializing a TFRecords file
    :param filename:
    :return:
    """
    proc = ManaProcessor169()
    global input_df
    input_df = temp_df = pd.read_csv(filename)
    data_column = temp_df[temp_df.columns[-1]]
    data_column = data_column.map(lambda x: str(x).replace('\t', ' ').replace('\n', ' ').lower())
    return proc._create_examples(data_column, 'test')

def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    processors = {
        "mana169": ManaProcessor169,
    }

    tokenization.validate_case_matches_checkpoint(DO_LOWER_CASE, INIT_CKPT)

    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)

    if MAX_SEQ_LENGTH > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (MAX_SEQ_LENGTH, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(OUTPUT_DIR)

    task_name = 'mana169'

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

    tpu_cluster_resolver = None

    hooks = []
    # create a logging tensor hook because this takes forever on cpu
    logger = tf.train.LoggingTensorHook({"Input" : "IteratorGetNext:0"}, every_n_iter=1)
    hooks.append(logger)
    # debug_hook = tfdbg.LocalCLIDebugHook()
    # hooks.append(debug_hook)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=OUTPUT_DIR,
        save_checkpoints_steps=1)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=INIT_CKPT,
        learning_rate=5e-5,
        num_train_steps=None,
        num_warmup_steps=None,
        use_tpu=False,
        use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=PREDICT_BATCH_SIZE)

    input_file = sys.argv[1]

    predict_examples = read_input_examples(input_file)
    num_actual_predict_examples = len(predict_examples)
    # if FLAGS.use_tpu:
    #     # TPU requires a fixed batch size for all batches, therefore the number
    #     # of examples must be a multiple of the batch size, or else examples
    #     # will get dropped. So we pad with fake examples which are ignored
    #     # later on.
    #     while len(predict_examples) % FLAGS.predict_batch_size != 0:
    #         predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(OUTPUT_DIR, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            MAX_SEQ_LENGTH, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", PREDICT_BATCH_SIZE)

    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn, hooks=hooks)

    output_predict_file = os.path.join(
        OUTPUT_DIR, sys.argv[2])

    scores_list = []

    num_written_lines = 0
    tf.logging.info("***** Predict results *****")
    for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
            break
        # writer.write(output_line)
        scores_list.append(probabilities)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples

    scores_array = np.array(scores_list)
    # write the scores in a useful form
    top3_scores = []
    all_topics = processor.get_labels()
    for i, row in enumerate(scores_array):
        top3_indices = row.argsort()[::-1][:3]
        # index 1, score 1, index 2, score 2, etc
        l = []
        l += [input_df.values[i][j] for j in range(input_df.shape[1] - 1)] # all but the original input
        l.append(str(input_df.values[i][-1]).replace('\n', '')) # take the original input and remove newlines
        for v in top3_indices:
            l.append(all_topics[v])
            l.append(row[v])
        top3_scores.append(l)

    score_df = pd.DataFrame(top3_scores, columns=list(input_df.columns.values) + ["Class 1", "Score 1", "Class 2", "Score 2", "Class 3", "Score 3"])
    score_df.to_csv(output_predict_file, index=None)

if __name__ == '__main__':
    tf.app.run()
