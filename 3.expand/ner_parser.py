import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description='for cnc ner')

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--if_load_pretrain", type=bool, default=False)

    # bi_lstm parameter
    parser.add_argument("--lstm_dim", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=4)
    parser.add_argument("--lstm_dropout", type=float, default=0.2)
    # crf parameter
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_class", type=int, default=8)
    parser.add_argument("--num_tag", type=int, default=17)

    return parser.parse_args()
