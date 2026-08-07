from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator, get_default_model_and_tokenizer
import pandas as pd
import pickle
import argparse


def run_rxnfp(output_filename, input_filename, label):
    df = pd.read_csv(input_filename)
    rxns = df[label].values
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    # df[label].values is a numpy array and the tokenizer rejects those outright
    # ("Input [...] is not valid. Should be a string, a list/tuple of strings"),
    # which happened for a single reaction too -- a type problem, not a batching
    # one, so list() is the actual fix. convert_batch then returns one 256-d
    # vector per row; convert returns a single flat 256-d vector that would only
    # fit a 256-row frame. The two agree exactly for one reaction.
    fps = rxnfp_generator.convert_batch(list(rxns))
    df['rxnfp'] = fps
    with open(output_filename, 'wb') as file:
        pickle.dump(df, file)
        
def parse_args():
    parser = argparse.ArgumentParser(description="Run rxnfp on a dataset")
    parser.add_argument('-out', '--out', required=True, help='Path to the output directory')
    parser.add_argument('-input', '--input', type=str, required=True, help='path to the dataframe')
    parser.add_argument('-label', '--label', type=str, required=True, help='label of the column')
    return parser.parse_args()

def main():
    args = parse_args()
    run_rxnfp(args.out, args.input, args.label)

main()