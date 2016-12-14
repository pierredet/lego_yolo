import tensorflow as tf
import sys

def main(path):
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(path)
    new_saver.restore(sess, tf.train.latest_checkpoint('./darkflow/ckpt'))
    # all_vars = tf.trainable_variables()
    truc = tf.get_collection('loss')[0]
    import pdb; pdb.set_trace()
    all_vars = tf.global_variables()
    for v in all_vars:
        print(v.name)

if __name__ == "__main__":
    main(sys.argv[1])
