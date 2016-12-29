import cPickle as pkl
import tensorflow as tf
import argparse
import os


def main(ckpt_path, output="weight.pkl"):
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(ckpt_path)
    new_saver.restore(sess, tf.train.latest_checkpoint('darkflow/ckpt/'))
    all_vars = tf.global_variables()
    [var for var in all_vars if "rmsprop" not in var.name.lower()]
    if os.path.exists(output):
        print "output path", output, 'already exists ! Please rename'
    with open(output, "wb") as f:
        pkl.dump([var.eval() for var in all_vars], f)
        f.flush()
        f.close()


def parse_args():
    parser = argparse.ArgumentParser("Dumps darkflow weights to a pickle file")

    parser.add_argument("ckpt_path",
                        help="path to a checkpoint from darkflow")
    parser.add_argument("-o", "--output",
                        help="npath to the binary pickle file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.output:
        main(args.ckpt_path, output=args.output)
    else:
        main(args.ckpt_path)
