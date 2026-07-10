import os
import pickle as pkl
import torch
# CUDA setup
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm
# Also a cross attention with the regression heads as well
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# Read in the embedded datasets and each of the processed datasets from CARE (we expect you to just download the task1 datasets and put them in processed datasets)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#import sysde
import argparse
from sklearn.metrics import average_precision_score, f1_score


# ------------------- Fine tuning code


def build_idxs_reaction_finetune(ec_level, rxn_to_embedding, train_df, reaction_cols, reaction_feature_df,
                        reaction_to_substrate, reaction_to_product):
    """
    Build index mappings for reaction data.
    """
    idx_to_embedding, value_to_index, idx_to_label, idx_to_entry = {}, {}, {}, {}
    idx_to_substrate, idx_to_product = {}, {}
    idx_to_mask, idx_to_labels = {}, {}
    class_similar_indicies = defaultdict(list)
    class_to_indicies = defaultdict(list)
    feat_map = {feat: dict(zip(reaction_feature_df['Reaction'], reaction_feature_df[feat].values)) for feat in reaction_cols}

    for i, entry, ec in train_df[['reaction_id', 'Reaction', 'EC number']].values:
        embedding = rxn_to_embedding.get(entry)
        if embedding is not None:
            idx_to_embedding[i] = torch.tensor(np.array(embedding, dtype='float32').flatten(), dtype=torch.float32)
            idx_to_substrate[i] = torch.tensor(np.array(reaction_to_substrate.get(entry), dtype='float32').flatten(), dtype=torch.float32)
            idx_to_product[i] = torch.tensor(np.array(reaction_to_product.get(entry), dtype='float32').flatten(), dtype=torch.float32)
            idx_to_entry[i] = entry
            value_to_index[entry] = i

            values, masks = [], []
            for feat in reaction_cols:
                val = feat_map[feat].get(entry)
                if val and 0.0 < val < 1.0:
                    values.append(val)
                    masks.append(1.0)
                else:
                    values.append(0.0)
                    masks.append(0)
            idx_to_mask[i] = torch.tensor(masks, dtype=torch.float32)
            idx_to_labels[i] = torch.tensor(values, dtype=torch.float32)

            class_number = '.'.join(ec.split('.')[:ec_level])
            idx_to_label[i] = class_number
            class_to_indicies[class_number].append(i)
    return (idx_to_embedding, idx_to_label, value_to_index, class_to_indicies,
            class_to_indicies, class_to_indicies, class_to_indicies,
            idx_to_mask, idx_to_labels, idx_to_substrate, idx_to_product)
    
def build_idxs_finetune(ec_level, seq_to_embedding, train_df, enzyme_cols, enzyme_feature_df):
    """
    Build index mappings for enzyme data.
    """
    idx_to_embedding, value_to_index, idx_to_label, idx_to_entry = {}, {}, {}, {}
    idx_to_mask, idx_to_labels = {}, {}
    class_similar_indicies = defaultdict(list)
    class_medium_indicies = defaultdict(list)
    class_easy_indicies = defaultdict(list)
    class_to_indicies = defaultdict(list)
    feat_map = {feat: dict(zip(enzyme_feature_df.Entry, enzyme_feature_df[feat].values)) for feat in enzyme_cols}

    for i, entry, ec in train_df[['protein_id', 'Entry', 'EC number']].values:
        embedding = seq_to_embedding.get(entry)
        if embedding is not None:
            idx_to_embedding[i] = torch.tensor(embedding.flatten(), dtype=torch.float32)
            idx_to_entry[i] = entry
            value_to_index[entry] = i
            class_number = '.'.join(ec.split('.')[:ec_level])
            idx_to_label[i] = class_number
            class_to_indicies[class_number].append(i)

            values, masks = [], []
            for feat in enzyme_cols:
                val = feat_map[feat].get(entry)
                if feat not in ['cofactor_encodings', 'subcellualr_encodings', 'active_encodings', 'ec_encodings']:
                    if val and 0.0 < val < 1.0:
                        values.append(val)
                        masks.append(1)
                    else:
                        values.append(0.0)
                        masks.append(0)
            idx_to_mask[i] = torch.tensor(masks, dtype=torch.float32)
            idx_to_labels[i] = torch.tensor(values, dtype=torch.float32)

            # Class groupings
            class_similar_indicies['.'.join(ec.split('.')[:ec_level - 1])].append(i)
    return (idx_to_embedding, idx_to_label, value_to_index, class_to_indicies,
            class_similar_indicies, class_similar_indicies, class_similar_indicies,
            idx_to_mask, idx_to_labels)


def build_paired_balanced_train_test_df_finetune(df, idx_to_label_reaction, class_to_indicies_protein, class_to_indicies_reaction, class_similar_indicies_reaction, ec_level=4, num_pos_samples=1000):
    """
    Build paired positive/negative samples for triplet-like training.
    """
    all_pair_embeddings, all_labels = [], []
    for reaction_id, protein_id, activity in df[['reaction_id', 'protein_id', 'activity']].values:
            # Positive and negative pairs
            all_pair_embeddings.extend([
                [protein_id, reaction_id],
                [protein_id, reaction_id],
                [protein_id, reaction_id],
                [protein_id, reaction_id]
            ])
            all_labels.extend([activity, activity, activity, activity])
            
    return all_pair_embeddings, all_labels
    
def create_dataset_finetune(protein_embedding, protein_train_df, enzyme_cols, protein_feature_df, reaction_to_embedding, reaction_train_df, reaction_cols, reaction_feature_df, reaction_to_substrate, reaction_to_product, num_pairs):
    idx_to_embedding_protein, idx_to_label_protein, value_to_index, class_to_indicies_protein, class_similar_indicies, class_medium_indicies, class_easy_indicies, protein_mask, protein_labels = build_idxs_finetune(4, protein_embedding, protein_train_df, enzyme_cols, protein_feature_df)
    idx_to_embedding_reaction, idx_to_label_reaction, value_to_index, class_to_indicies_reaction, class_similar_indicies_reaction, class_medium_indicies_reaction, class_easy_indicies_reaction, reaction_mask, reaction_labels, substrate_idx, product_idx = build_idxs_reaction_finetune(4, reaction_to_embedding, reaction_train_df, reaction_cols, reaction_feature_df, reaction_to_substrate, reaction_to_product)    
    all_pair_embeddings, all_labels = build_paired_balanced_train_test_df_finetune(protein_train_df, 
                                                idx_to_label_reaction,
                                                class_to_indicies_protein, 
                                                class_to_indicies_reaction,
                                                class_similar_indicies_reaction, ec_level=4, num_pos_samples=num_pairs)
    all_labels = torch.tensor(all_labels, dtype=torch.float32)
    dataset = EnzymePointerDataset(idx_to_embedding_protein, idx_to_embedding_reaction, substrate_idx, product_idx, protein_labels, protein_mask, reaction_labels, reaction_mask, all_pair_embeddings, all_labels) #[0], all_labels[1])
    return dataset


def average_retransformed_features(df_to_add, dfs, enzyme_cols, reaction_cols, label=''):
    for enzyme_col in enzyme_cols + reaction_cols:
        data = []
        for d in dfs:
            data.append(d[f'inverse_transformed_pred_{enzyme_col}'].values)
        data = np.array(data)
        df_to_add[f'{label}{enzyme_col}_mean'] = np.mean(data, axis=0)
        df_to_add[f'{label}{enzyme_col}_std'] = np.std(data, axis=0)

    return df_to_add
    
    
    
def predict_on_dataset_ensemble(label, df, models, protein_embedding_column, product_embedding_column, 
                                substrate_embedding_column, reaction_embedding_column, enzyme_feature_scaler, 
                                reaction_feature_scaler,
                                action_column=None, seq_column=None, plot_fig=False):
    enzyme_cols = ['Length', 'Mass', 'Polarity', 'temperature']
    reaction_cols = ['substrates_MolWt', 'substrates_MolLogP', 'substrates_MaxPartialCharge', 'substrates_MinPartialCharge', 
                 'products_MolWt', 'products_TPSA', 'products_MolLogP', 'products_MaxPartialCharge', 'products_MinPartialCharge']
    
    if not action_column:
        action_column = 'EmptyAction'
        df[action_column] = 0
    y_true = df[action_column].values

    torch.set_float32_matmul_precision('high')
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Collect predictions from all models
    all_model_preds = []
    validation_dfs = []
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            for model in models:
                model.eval()
                for module in model.modules():
                    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                        module.eval()
                        module.track_running_stats = False
                        module.running_mean = module.running_mean.detach()
                        module.running_var = module.running_var.detach()
                model.to(device)
                model.eval()

                X_batch_enzyme = torch.tensor(df[protein_embedding_column].values.tolist()).to(device)
                X_batch_product = torch.tensor(df[product_embedding_column].values.tolist()).to(device)
                X_batch_substrate = torch.tensor(df[substrate_embedding_column].values.tolist()).to(device)
                X_batch_reaction = torch.tensor(df[reaction_embedding_column].values.tolist()).to(device)

                pred = model(X_batch_enzyme, X_batch_product, X_batch_substrate, X_batch_reaction)

                # Sigmoid since do this in the loss normally
                pred = pred.cpu()
                probs = torch.sigmoid(pred[:, 0].squeeze())
                pred[:, 0] = probs.float()

                validation_df = retransform_scaled_predictions(df, pred, enzyme_feature_scaler, enzyme_cols, 
                                                       reaction_feature_scaler, reaction_cols)
                all_model_preds.append(pred.numpy())
                validation_dfs.append(validation_df.copy())

    all_model_preds = np.stack(all_model_preds, axis=0)
    print('ALL', all_model_preds.shape)

    # Calc mean and variance 
    mean_preds = np.mean(all_model_preds, axis=0)
    std_preds = np.std(all_model_preds, axis=0)

    # Classification metrics
    y_pred = np.round(mean_preds[:, 0])
    acc = np.mean(y_pred == y_true)
    f1 = f1_score(y_true, y_pred, average='macro')
    auprc = average_precision_score(y_true, y_pred)

    print(f"Different true values: {np.unique(y_true)}, y_true.sum: {y_true.sum()}")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUPRC: {auprc}")
    return mean_preds, std_preds, f1, acc, validation_dfs



seed = 118
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Turn off auto-optimization

np.random.seed(seed)
random.seed(seed)
torch.set_default_dtype(torch.float32)  # Set precision to 32-bit (default)



def retransform_scaled_predictions(test, pred, enzyme_feature_scaler, enzyme_cols, reaction_feature_scaler, reaction_cols):
    # Test how the test set accuracy is, also check what the length looks like for this prediction!
    pred_enzyme_cols = []
    for i, v in enumerate(enzyme_cols):
        test[f'pred_{v}'] = pred[:, i+1].detach()
        pred_enzyme_cols.append(f'pred_{v}')
        
    scaled_cols = enzyme_feature_scaler.inverse_transform(test[pred_enzyme_cols].values)
    for i, v in enumerate(pred_enzyme_cols):
        test[f'inverse_transformed_{v}'] = scaled_cols[:, i]
    
    pred_reaction_cols = []
    # Do the same for the reactions
    for i, v in enumerate(reaction_cols):
        test[f'pred_{v}'] = pred[:, i+1+len(pred_enzyme_cols)].detach()
        pred_reaction_cols.append(f'pred_{v}')
        
    scaled_cols = reaction_feature_scaler.inverse_transform(test[pred_reaction_cols].values)
    for i, v in enumerate(pred_reaction_cols):
        test[f'inverse_transformed_{v}'] = scaled_cols[:, i]
        
    test['pred_Activity'] = pred[:, 0].detach()
    return test

    
def save(model, reaction_feature_scaler, enzyme_feature_scaler, config, optimizer, output_name, output_dir):
    """ Save a pytorch model """ 
    checkpoint = { 
        'model': model,
        'optimizer': optimizer
    }
    torch.save(checkpoint,  os.path.join(output_dir, f'{output_name}_checkpoint.pth'))

    torch.save(model.state_dict(), os.path.join(output_dir, f'{output_name}_torch.pkl'))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, f'{output_name}_optimizer.pkl'))

    # Save everything else as a dict
    saving_dict = {'enzyme_feature_scaler': enzyme_feature_scaler,
                   'reaction_feature_scaler': reaction_feature_scaler,

                   'config': config}
    pickle.dump(saving_dict, open(os.path.join(output_dir, f'{output_name}_conf.pkl'), 'wb'))
    
    print('Saved torch model to:', os.path.join(output_dir, f'{output_name}_torch.pkl'), 
          '\n Saved scaler and config to: ', os.path.join(output_dir, f'{output_name}_conf.pkl'))
    
def load(config_path, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    config = pickle.load(open(config_path, 'rb'))
    return model, config, optimizer
    
class NeuralNetworkModelWithAttention(nn.Module):
    
    def __init__(self, output_dim, config):
        super(NeuralNetworkModelWithAttention, self).__init__()
        layers = config['layers']
        attention = config['attention']
        self.config = config
        self.output_dim = output_dim
        dropout = config['dropout']
        num_heads = config['num_heads']
        subtype = config['subtype']
        self.subtype = subtype
        self.dropout = nn.Dropout(dropout)

        self.forward_enzyme = nn.Linear(attention['enzyme_size'], attention['embed_size'])
        self.forward_reaction = nn.Linear(attention['reaction_size'], attention['embed_size'])
        self.forward_substrate = nn.Linear(attention['substrate_size'], attention['embed_size'])
        self.forward_product = nn.Linear(attention['product_size'], attention['embed_size'])

        self.norm_enzyme = nn.LayerNorm(attention['embed_size'])
        self.dropout_enzyme = nn.Dropout(dropout)

        self.norm_reaction = nn.LayerNorm(attention['embed_size'])
        self.dropout_reaction = nn.Dropout(dropout)

        self.norm_substrate = nn.LayerNorm(attention['embed_size'])
        self.dropout_substrate = nn.Dropout(dropout)

        self.norm_product = nn.LayerNorm(attention['embed_size'])
        self.dropout_product = nn.Dropout(dropout)

        self.attention_enzyme_reaction = CrossAttention(attention['embed_size'], attention['enzyme_size'], attention['reaction_size'])
        self.attention_reaction_enzyme = CrossAttention(attention['embed_size'], attention['reaction_size'], attention['enzyme_size'])
        
        self.attention_enzyme_substrate = CrossAttention(attention['embed_size'], attention['enzyme_size'], attention['substrate_size'])
        self.attention_substrate_enzyme = CrossAttention(attention['embed_size'], attention['substrate_size'], attention['enzyme_size'])
        
        self.input_layer = nn.Linear(attention['embed_size'] + attention['embed_size'] + attention['embed_size'] + attention['embed_size'] + attention['embed_size'], layers[0])        

        self.self_attention = nn.Linear(attention['embed_size'] + attention['embed_size'] + attention['embed_size'] + attention['embed_size'], attention['embed_size'])
    
        self.layer0_norm = nn.LayerNorm(layers[0])
        self.dropout0 = nn.Dropout(dropout)

        self.self_attention_norm = nn.LayerNorm(attention['embed_size'])
        self.self_attention_dropout = nn.Dropout(dropout)
        
        self.hidden_layer1  = nn.Linear(layers[0], layers[1])
        self.layer1_norm = nn.LayerNorm(layers[1])
        self.dropout1 = nn.Dropout(dropout)

        self.hidden_layer2  = nn.Linear(layers[1], layers[2])
        self.layer2_norm = nn.LayerNorm(layers[2])
        self.dropout2 = nn.Dropout(dropout)
        
        self.output_layer = nn.Linear(layers[2], output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, enzyme, product, substrate, reaction):

        enzyme_out = self.relu(self.norm_enzyme(self.dropout_enzyme(self.forward_enzyme(enzyme))))
        reaction_out = self.relu(self.norm_reaction(self.dropout_reaction(self.forward_reaction(reaction))))
        substrate_out = self.relu(self.norm_substrate(self.dropout_substrate(self.forward_substrate(substrate))))
        product_out = self.relu(self.norm_product(self.dropout_product(self.forward_product(product))))

        attn_out = torch.cat((substrate_out, reaction_out, enzyme_out, product_out), 1)
        
        out_1 = self.attention_enzyme_reaction(enzyme, reaction)
        out_2 = self.attention_reaction_enzyme(reaction, enzyme)   
        out_3 = self.attention_enzyme_substrate(enzyme, substrate)
        out_4 = self.attention_substrate_enzyme(substrate, enzyme)   

        cross_attn_output = (out_1 + out_2 + out_3 + out_4)/4
        
        # take the mean of cross attention
        concat_input = torch.cat((attn_out, cross_attn_output), 1)
        out =  self.relu(self.layer0_norm(self.dropout0(self.input_layer(concat_input))))
        out =  self.relu(self.layer1_norm(self.dropout1(self.hidden_layer1(out))))
        out =  self.relu(self.layer2_norm(self.dropout2(self.hidden_layer2(out))))
        out =  self.output_layer(out)
        
        return out

    def calculate_loss(self, outputs, targets, protein_mask, reaction_mask, protein_labels, reaction_labels):#, masks, start=0, end=-1):
        activity_loss = nn.functional.binary_cross_entropy_with_logits(outputs[:, 0], targets.squeeze())
        avg_loss = 1 # Have an offset incase there are no values to add (will be max 1/14)
        j = 0
        for i in range(0, self.output_dim - 1):
            if i < protein_mask.shape[1]:
                tmp_avg_loss = self.mse_loss(outputs[:, i + 1], protein_labels[:, i].squeeze())
                # Multiply by the mask
                tmp_avg_loss = tmp_avg_loss*protein_mask[:, i]
                avg_loss += tmp_avg_loss.sum() / (protein_mask[:, i].sum() + 1)
            else:
                tmp_avg_loss = self.mse_loss(outputs[:, i + 1], reaction_labels[:, j].squeeze())
                tmp_avg_loss = tmp_avg_loss*reaction_mask[:, j]
                avg_loss += tmp_avg_loss.sum() / (reaction_mask[:, j].sum() + 1)
                j += 1
        # Multiply by the mask
        avg_loss = avg_loss/self.output_dim
        activity_loss = activity_loss.float()
        avg_loss = avg_loss.float()
        loss = activity_loss + avg_loss
        return loss


class CrossAttention(nn.Module):
    # Adapted from:  https://github.com/GENTEL-lab/EnzymeCAGE/blob/master/enzymecage/interaction.py
    def __init__(self, output_dim, query_input_dim, key_input_dim):
        super(CrossAttention, self).__init__()        
        self.out_dim = output_dim
        self.W_Q = nn.Linear(query_input_dim, output_dim)
        self.W_K = nn.Linear(key_input_dim, output_dim)
        self.W_V = nn.Linear(key_input_dim, output_dim)
        self.scale_val = self.out_dim ** 0.5
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query_input, key_input, attn_bias=None):
        query = self.W_Q(query_input)
        key = self.W_K(key_input)
        value = self.W_V(key_input)
        
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / self.scale_val
        
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias

        attn_weights = self.softmax(attn_weights)
        output = torch.matmul(attn_weights, value)
        
        return output


def get_mask_y_map(training_df, enzyme_cols, reaction_cols, enzyme_feature_df, reaction_feature_df):
    mask_map = [torch.FloatTensor(np.ones(len(training_df))).reshape(-1, 1)]
    y_map = [torch.FloatTensor(np.array(training_df['Action'].values)).reshape(-1, 1)]

    for feat in enzyme_cols:
        # Get the 
        values = []
        masks = []
        feat_map = dict(zip(enzyme_feature_df.Entry, enzyme_feature_df[feat].values))
        for entry in training_df['Entry'].values:
            val = feat_map.get(entry)
            if feat not in ['cofactor_encodings', 'subcellualr_encodings', 'active_encodings', 'ec_encodings']:
                if val and val > 0.0 and val < 1.0:
                    values.append(val)
                    masks.append(1)
                else:
                    values.append(0.0)
                    masks.append(0)
        mask_map.append(torch.FloatTensor(np.array(masks)).reshape(-1, 1))
        y_map.append(torch.FloatTensor(np.array(values)).reshape(-1, 1))
        
    for feat in reaction_cols:
        values = []
        masks = []
        feat_map = dict(zip(reaction_feature_df['Reaction'], reaction_feature_df[feat].values))
        for entry in training_df['Reaction'].values:
            val = feat_map.get(entry)
            if val and val > 0.0 and val < 1.0:
                values.append(val)
                masks.append(1.0)
            else:
                values.append(0.0)
                masks.append(0)
        mask_map.append(torch.FloatTensor(np.array(masks)).reshape(-1, 1))
        y_map.append(torch.FloatTensor(np.array(values)).reshape(-1, 1))
    return y_map, mask_map
    
def build_idxs(ec_level, seq_to_embedding, train_df, enzyme_cols, enzyme_feature_df):
    """
    Build index mappings for enzyme data.
    """
    idx_to_embedding, value_to_index, idx_to_label, idx_to_entry = {}, {}, {}, {}
    idx_to_mask, idx_to_labels = {}, {}
    class_similar_indicies = defaultdict(list)
    class_medium_indicies = defaultdict(list)
    class_easy_indicies = defaultdict(list)
    class_to_indicies = defaultdict(list)
    feat_map = {feat: dict(zip(enzyme_feature_df.Entry, enzyme_feature_df[feat].values)) for feat in enzyme_cols}

    for i, (entry, ec) in enumerate(train_df[['Entry', 'EC number']].values):
        embedding = seq_to_embedding.get(entry)
        if embedding is not None:
            idx_to_embedding[i] = torch.tensor(embedding.flatten(), dtype=torch.float32)
            idx_to_entry[i] = entry
            value_to_index[entry] = i
            class_number = '.'.join(ec.split('.')[:ec_level])
            idx_to_label[i] = class_number
            class_to_indicies[class_number].append(i)

            values, masks = [], []
            for feat in enzyme_cols:
                val = feat_map[feat].get(entry)
                if feat not in ['cofactor_encodings', 'subcellualr_encodings', 'active_encodings', 'ec_encodings']:
                    if val and 0.0 < val < 1.0:
                        values.append(val)
                        masks.append(1)
                    else:
                        values.append(0.0)
                        masks.append(0)
            idx_to_mask[i] = torch.tensor(masks, dtype=torch.float32)
            idx_to_labels[i] = torch.tensor(values, dtype=torch.float32)

            # Class groupings
            class_similar_indicies['.'.join(ec.split('.')[:ec_level - 1])].append(i)
            class_medium_indicies['.'.join(ec.split('.')[:ec_level - 2])].append(i)
            class_easy_indicies['.'.join(ec.split('.')[:ec_level - 3])].append(i)
    return (idx_to_embedding, idx_to_label, value_to_index, class_to_indicies,
            class_similar_indicies, class_medium_indicies, class_easy_indicies,
            idx_to_mask, idx_to_labels)

def build_idxs_reaction(ec_level, rxn_to_embedding, train_df, reaction_cols, reaction_feature_df,
                        reaction_to_substrate, reaction_to_product):
    """
    Build index mappings for reaction data.
    """
    idx_to_embedding, value_to_index, idx_to_label, idx_to_entry = {}, {}, {}, {}
    idx_to_substrate, idx_to_product = {}, {}
    idx_to_mask, idx_to_labels = {}, {}
    class_similar_indicies = defaultdict(list)
    class_medium_indicies = defaultdict(list)
    class_easy_indicies = defaultdict(list)
    class_to_indicies = defaultdict(list)
    feat_map = {feat: dict(zip(reaction_feature_df['Reaction'], reaction_feature_df[feat].values)) for feat in reaction_cols}

    for i, (entry, ec) in enumerate(train_df[['Reaction', 'EC number']].values):
        embedding = rxn_to_embedding.get(entry)
        if embedding is not None:
            idx_to_embedding[i] = torch.tensor(np.array(embedding, dtype='float32').flatten(), dtype=torch.float32)
            idx_to_substrate[i] = torch.tensor(np.array(reaction_to_substrate.get(entry), dtype='float32').flatten(), dtype=torch.float32)
            idx_to_product[i] = torch.tensor(np.array(reaction_to_product.get(entry), dtype='float32').flatten(), dtype=torch.float32)
            idx_to_entry[i] = entry
            value_to_index[entry] = i

            values, masks = [], []
            for feat in reaction_cols:
                val = feat_map[feat].get(entry)
                if val and 0.0 < val < 1.0:
                    values.append(val)
                    masks.append(1.0)
                else:
                    values.append(0.0)
                    masks.append(0)
            idx_to_mask[i] = torch.tensor(masks, dtype=torch.float32)
            idx_to_labels[i] = torch.tensor(values, dtype=torch.float32)

            class_number = '.'.join(ec.split('.')[:ec_level])
            idx_to_label[i] = class_number
            class_to_indicies[class_number].append(i)
            class_similar_indicies['.'.join(ec.split('.')[:ec_level - 1])].append(i)
            class_medium_indicies['.'.join(ec.split('.')[:ec_level - 2])].append(i)
            class_easy_indicies['.'.join(ec.split('.')[:ec_level - 3])].append(i)
    return (idx_to_embedding, idx_to_label, value_to_index, class_to_indicies,
            class_similar_indicies, class_medium_indicies, class_easy_indicies,
            idx_to_mask, idx_to_labels, idx_to_substrate, idx_to_product)


def build_paired_balanced_train_test_df(idx_to_label_reaction, class_to_indicies_protein, class_to_indicies_reaction, class_similar_indicies_reaction, ec_level=4, num_pos_samples=1000):
    """
    Build paired positive/negative samples for triplet-like training.
    """
    all_ecs = list(set(class_to_indicies_protein) & set(class_to_indicies_reaction))
    all_pair_embeddings, all_labels = [], []
    for _ in tqdm(range(num_pos_samples)):
        ec = random.choice(all_ecs)
        protein = random.choice(class_to_indicies_protein[ec])
        reaction = random.choice(class_to_indicies_reaction[ec])
        similar_group = class_similar_indicies_reaction.get('.'.join(ec.split('.')[:ec_level - 1]), [])
        if not similar_group:
            continue
        reaction_similar = random.choice(similar_group)
        reaction_ec = idx_to_label_reaction.get(reaction_similar)
        if reaction_ec != ec and class_to_indicies_protein.get(reaction_ec) and class_to_indicies_reaction.get(reaction_ec):
            sample_i = random.choice(class_to_indicies_protein[reaction_ec])
            # Positive and negative pairs of similar ones that we put together
            all_pair_embeddings.extend([
                [protein, reaction],
                [protein, reaction_similar],
                [sample_i, reaction_similar],
                [sample_i, reaction]
            ])
            all_labels.extend([1, 0, 1, 0])
    print(all_labels[0:3])
    return all_pair_embeddings, all_labels


# Setup for contastive loss
class EnzymePointerDataset(Dataset):
    def __init__(self, protein, reaction, substrate, product, protein_labels, protein_mask, reaction_labels, reaction_mask, pair_indices, labels): #, labels2):
        self.protein = protein     # Dict - e.g.  {i: torch.rand(512) for i in range(1000)}
        self.substrate = substrate
        self.product = product
        self.protein_labels = protein_labels
        self.protein_mask = protein_mask
        self.reaction = reaction
        self.reaction_labels = reaction_labels
        self.reaction_mask = reaction_mask 
        self.pair_indices = pair_indices # Each pair is a tuple of keys from `main_store`, like (0, 5) or (2, 8)
        self.labels = labels

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        # Retrieve indices for the current pair
        idx1, idx2= self.pair_indices[idx]
        # Look up the actual data in the main store using these indices
        enzyme, reaction, substrate, product = self.protein[idx1], self.reaction[idx2], self.substrate[idx2], self.product[idx2]
        protein_labels, reaction_labels, protein_mask, reaction_mask = self.protein_labels[idx1], self.reaction_labels[idx2], self.protein_mask[idx1], self.reaction_mask[idx2]
        # (enzyme, reaction, substrate, product, protein_mask, reaction_mask, protein_labels, reaction_labels, labels)
        return enzyme, reaction, substrate, product, protein_mask, reaction_mask, protein_labels, reaction_labels, self.labels[idx]

  
def train_model_with_cross_attention(model, optimizer, loss_fn, train_set, protein_val_set, config, early_stop=False):
    batch_size = config.get('batch_size') or 100
    num_epochs = config.get('num_epochs') or 10
    early_stop = config.get('early_stop') or 4
    patience = early_stop
    
    if not early_stop:
        early_stop = num_epochs
        patience = num_epochs

    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=30, pin_memory=True)
    val_loader = DataLoader(protein_val_set, batch_size=100, shuffle=False, num_workers=10, pin_memory=True)

    # History dictionary
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_protein_loss": [],
        "val_protein_acc": [],
        "val_reaction_loss": [],
        "val_reaction_acc": []
    }
    
    patience_counter = 0
    best_model_state = None
    best_val_loss = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        avg_acc = 0
        avg_train_loss = 0
        model.train()

        for enzyme, reaction, substrate, product, protein_mask, reaction_mask, protein_labels, reaction_labels, labels in tqdm(dataloader):
                # take a batch
                X_batch_enzyme = enzyme.to(device).float()  
                X_batch_reaction = reaction.to(device).float()
                X_batch_product = product.to(device).float()
                X_batch_substrate = substrate.to(device).float()
                protein_labels = protein_labels.to(device).float()
                reaction_labels = reaction_labels.to(device).float()
                y_batch = labels.to(device).float()
                protein_mask = protein_mask.to(device)
                reaction_mask = reaction_mask.to(device)
                # forward pass
                y_pred = model(X_batch_enzyme, X_batch_product, X_batch_substrate, X_batch_reaction)
                preds = (torch.sigmoid(y_pred[:, 0].squeeze()) > 0.5).float()
                avg_acc += (preds == y_batch).float().mean()
                loss = loss_fn(y_pred, y_batch, protein_mask, reaction_mask, protein_labels, reaction_labels)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                avg_train_loss += loss.item()
                # print progress
        model.eval()
        avg_acc = avg_acc / len(dataloader)
        val_loss = 0.0
        val_acc  = 0
        with torch.no_grad():
            for enzyme, reaction, substrate, product, protein_mask, reaction_mask, protein_labels, reaction_labels, labels in tqdm(val_loader):
                    # take a batch
                    X_batch_enzyme = enzyme.to(device).float()
                    X_batch_reaction = reaction.to(device).float()
                    X_batch_product = product.to(device).float()
                    X_batch_substrate = substrate.to(device).float()
                    protein_mask = protein_mask.to(device).float()
                    reaction_mask = reaction_mask.to(device).float()
                    y_batch = labels.to(device).float()
                    protein_labels = protein_labels.to(device)
                    reaction_labels = reaction_labels.to(device)
                    # forward pass
                    y_pred = model(X_batch_enzyme, X_batch_product, X_batch_substrate, X_batch_reaction)
                    preds = (torch.sigmoid(y_pred[:, 0]) > 0.5).float()
                    acc = (preds == y_batch).float().mean()
                    val_loss += loss_fn(y_pred, y_batch, protein_mask, reaction_mask, protein_labels, reaction_labels)
                    val_acc += acc
        avg_val_acc = val_acc / len(val_loader)
        avg_val_loss = val_loss/len(val_loader)
        
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Protein Loss: {avg_val_loss:.4f} " +
              f" Train Acc: {avg_acc:.4f}, Val Protein Acc: {avg_val_acc:.4f}" )

        # Update history
        history["train_loss"].append(float(avg_train_loss))
        history["train_acc"].append(float(avg_acc))
        history["val_protein_loss"].append(float(avg_val_loss))
        history["val_protein_acc"].append(float(avg_val_acc))
        
        # Early stopping
        if avg_val_acc > best_val_loss:
            best_val_loss = avg_val_acc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, optimizer, history



def create_dataset(ec_level, protein_embedding, protein_train_df, enzyme_cols, protein_feature_df, reaction_to_embedding, reaction_train_df, reaction_cols, reaction_feature_df, reaction_to_substrate, reaction_to_product, num_pairs):
    idx_to_embedding_protein, idx_to_label_protein, value_to_index, class_to_indicies_protein, class_similar_indicies, class_medium_indicies, class_easy_indicies, protein_mask, protein_labels = build_idxs(4, protein_embedding, protein_train_df, enzyme_cols, protein_feature_df)
    idx_to_embedding_reaction, idx_to_label_reaction, value_to_index, class_to_indicies_reaction, class_similar_indicies_reaction, class_medium_indicies_reaction, class_easy_indicies_reaction, reaction_mask, reaction_labels, substrate_idx, product_idx = build_idxs_reaction(4, reaction_to_embedding, reaction_train_df, reaction_cols, reaction_feature_df, reaction_to_substrate, reaction_to_product)    

    all_pair_embeddings, all_labels = build_paired_balanced_train_test_df(
                                                idx_to_label_reaction,
                                                class_to_indicies_protein, 
                                                class_to_indicies_reaction,
                                                class_similar_indicies_reaction, ec_level=4, num_pos_samples=num_pairs)
    if ec_level == 3 or ec_level == 2:
        all_pair_embeddings_1, all_labels_1 = build_paired_balanced_train_test_df(
                                        idx_to_label_reaction,
                                        class_to_indicies_protein, 
                                        class_to_indicies_reaction,
                                        class_medium_indicies_reaction, ec_level=3, num_pos_samples=num_pairs)
        all_pair_embeddings = np.concatenate([all_pair_embeddings, all_pair_embeddings_1], axis=0)
        all_labels = np.concatenate([all_labels, all_labels_1], axis=0)
    if ec_level == 2:
        all_pair_embeddings_1, all_labels_1 = build_paired_balanced_train_test_df(
                                        idx_to_label_reaction,
                                        class_to_indicies_protein, 
                                        class_to_indicies_reaction,
                                        class_easy_indicies_reaction, ec_level=2, num_pos_samples=num_pairs)
        all_pair_embeddings = np.concatenate([all_pair_embeddings, all_pair_embeddings_1], axis=0)
        all_labels = np.concatenate([all_labels, all_labels_1], axis=0)
        
    all_labels = torch.tensor(all_labels, dtype=torch.float32)
    dataset = EnzymePointerDataset(idx_to_embedding_protein, idx_to_embedding_reaction, substrate_idx, product_idx, protein_labels, protein_mask, reaction_labels, reaction_mask, all_pair_embeddings, all_labels) #[0], all_labels[1])
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train enzyme-reaction prediction model')
    parser.add_argument('--label', type=str, default='benchmark',
                      help='Label for the run.')
    parser.add_argument('--reaction_level', type=str, default='easy',
                      help='Level of reaction data to use (default: easy)')
    parser.add_argument('--protein_level', type=str, default='30',
                      help='Level of protein data to use (default: 30)')
    parser.add_argument('--ec_level', type=int, default=4,
                      help='EC number level to use (default: 4)')
    parser.add_argument('--data_dir', type=str, 
                      default='data/',
                      help='Directory containing input data')
    parser.add_argument('--train_dataset', type=str, 
                      default=None,
                      help='Subset of the protein training dataset')
    parser.add_argument('--gpu', type=str)
    return parser.parse_args()

def load_data(data_folder):
    protein_df = pd.read_csv(f'{data_folder}/protein.csv')
    reaction_df = pd.read_csv(f'{data_folder}/enzymemap_v2_brenda2023_reactions.csv')

    with open(f'{data_folder}/embeddings.pkl', 'rb') as fin:
        embeddings = pkl.load(fin)
        reaction_to_embedding = embeddings['reaction_to_rxnfp']
        reaction_to_substrate = embeddings['reaction_to_substrate_unimol']
        reaction_to_product = embeddings['reaction_to_product_unimol']
        uniprot_to_esm2 = embeddings['uniprot_to_esm2']
        uniprot_to_esm3 = embeddings['uniprot_to_esm3']
        reaction_to_all = embeddings['reaction_to_all']
        reaction_to_unimol = embeddings['reaction_to_unimol']


    feature_map = {}
    for reaction_feature in ['substrates_MolWt', 'products_MolWt', 'substrates_MinPartialCharge', 'products_MinPartialCharge', 'products_MolLogP', 'substrates_MolLogP']:
        feature_map[reaction_feature] = dict(zip(reaction_df['unmapped'], reaction_df[reaction_feature]))

    for protein_feature in ['Length', 'Mass', 'Polarity', 'temperature']:
        feature_map[protein_feature] = dict(zip(protein_df['Entry'], protein_df[protein_feature]))

    enzyme_cols = ['Length', 'Mass', 'Polarity', 'temperature']

    enzyme_feature_scaler = MinMaxScaler()
    protein_feature_df = protein_df.copy()
    protein_feature_df = protein_feature_df[protein_feature_df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]  # .astype(np.float64) ?
    protein_feature_df[enzyme_cols] = enzyme_feature_scaler.fit_transform(protein_feature_df[enzyme_cols])

    reaction_cols = ['substrates_MolWt', 'substrates_MolLogP', 'substrates_MaxPartialCharge', 'substrates_MinPartialCharge', 
                    'products_MolWt', 'products_TPSA', 'products_MolLogP', 'products_MaxPartialCharge', 'products_MinPartialCharge']

    reaction_feature_scaler = MinMaxScaler()
    reaction_feature_df = reaction_df.copy()

    reaction_feature_df = reaction_feature_df[reaction_feature_df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]  # .astype(np.float64) ?

    for col in reaction_cols:
        vals = reaction_feature_df[col].values
        sd = np.nanstd(vals)
        median = np.nanmean(vals)
        upper = median + 3*sd
        lower = median - 3*sd
        reaction_feature_df[col] = reaction_feature_df[col].clip(upper=upper, lower=lower)

    return uniprot_to_esm3, reaction_feature_df, reaction_feature_scaler, enzyme_cols, protein_feature_df, reaction_to_substrate, reaction_to_product, reaction_to_embedding, reaction_cols, enzyme_feature_scaler


def run(label, reaction_level, protein_level, ec_level, data_dir, train_dataset):
    # Load the data
    uniprot_to_esm3, reaction_feature_df, reaction_feature_scaler, enzyme_cols, protein_feature_df, reaction_to_substrate, reaction_to_product, reaction_to_embedding, reaction_cols, enzyme_feature_scaler = load_data(data_dir)

    validation_pairs = 2000
    num_pairs = 500000
    for reaction_level in ['easy', 'medium', 'hard']:
        reaction_feature_df[reaction_cols] = reaction_feature_scaler.fit_transform(reaction_feature_df[reaction_cols])
        reaction_train_df = pd.read_csv(f'{data_dir}/EC_protein-{protein_level}_reaction-{reaction_level}_train.csv')
        reaction_test_df = pd.read_csv(f'{data_dir}/EC_protein-{protein_level}_reaction-{reaction_level}_test.csv')
        if train_dataset:
            protein_train_df = pd.read_csv(train_dataset)
        else:
            protein_train_df = pd.read_csv(f'{data_dir}/enzyme_protein-{protein_level}_reaction-{reaction_level}_train-filtered.csv')
        protein_test_df = pd.read_csv(f'{data_dir}/enzyme_protein-{protein_level}_reaction-{reaction_level}_test.csv')
        for model_type in ['ESRP']:
            for ec_level in [1]: #, 2, 3, 4]:
                for model_idx in range(1, 2):
                    protein_embedding = uniprot_to_esm3
                    reaction_feature_df['Reaction'] = reaction_feature_df['unmapped'].values
                    train_dataset = create_dataset(ec_level, protein_embedding, protein_train_df, enzyme_cols, protein_feature_df, 
                                reaction_to_embedding, reaction_train_df, reaction_cols, reaction_feature_df, reaction_to_substrate, reaction_to_product, num_pairs)    
                    # Use the test dataset for validation
                    protein_valid_dataset = create_dataset(ec_level, protein_embedding, protein_test_df, enzyme_cols, protein_feature_df, 
                                reaction_to_embedding, reaction_test_df, reaction_cols, reaction_feature_df, reaction_to_substrate, reaction_to_product, validation_pairs)    

                    config = {
                            'layers': [512, 256, 128, 0, 0], 
                            'dropout': 0.2,
                            'num_epochs': 2,
                            'batch_size': 10000,
                            'output_dim': 14,
                            'learning_rate': 0.0001,
                            'num_heads': 8,
                            'early_stop': 5,
                            'attention': {'product_size': 768, 
                                            'substrate_size': 768, 
                                            'reaction_size': 256,
                                            'enzyme_size': 1536, 
                                            'embed_size': 1024, 
                                            'attention_type': 'cross'}}
                    
                    config['subtype'] = model_type
                    
                    model = NeuralNetworkModelWithAttention(config.get('output_dim'), config)
                    
                    # creating our optimizer and loss function object
                    loss_fn = model.calculate_loss
                    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate'))
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device)
                    
                    model, optimizer, history = train_model_with_cross_attention(model, optimizer, loss_fn, train_dataset, protein_valid_dataset, config)
                    
                    history_path = os.path.join(f'{data_dir}/trained_models_new', f'{label}_{reaction_level}_{protein_level}_{model_type}_{ec_level}_model_{model_idx}_{num_pairs}_history.pkl')
                    with open(history_path, 'wb') as f:
                        pkl.dump(history, f)
                        
                    save(model, reaction_feature_scaler,  enzyme_feature_scaler, config, optimizer, f'{label}_{reaction_level}_{protein_level}_{model_type}_{ec_level}_model_{model_idx}_{num_pairs}', f'{data_dir}trained_models_new/')

    
def main():
    args = parse_args()
    run(args.label, args.reaction_level, args.protein_level, args.ec_level, args.data_dir, args.train_dataset)

if __name__ == '__main__':
    main()
# Run 
# nohup python run_ml_04092025.py --reaction_level easy --protein_level 0-50 --ec_level 4 --gpu 0 & # 3095830
# nohup python run_ml_04092025.py --reaction_level easy --protein_level 0-50 --ec_level 3 --gpu 0 & # 3095830
# nohup python run_ml_04092025.py --reaction_level easy --protein_level 0-50 --ec_level 2 --gpu 0 & # 3095830
# nohup python run_ml_04092025.py --reaction_level easy --protein_level 0-50 --ec_level 1 --gpu 1 & 
# nohup python run_ml.py --reaction_level medium --protein_level 0-50 --ec_level 4 --gpu 1 & # 3095830
# nohup python run_ml.py --reaction_level medium --protein_level 0-50 --ec_level 3 --gpu 1 & # 3095830
# nohup python run_ml.py --reaction_level medium --protein_level 0-50 --ec_level 2 --gpu 1 & # 3095830

# nohup python run_ml.py --reaction_level hard --protein_level 0-50 --ec_level 4 --gpu 0 & # 3095830
# nohup python run_ml.py --reaction_level hard --protein_level 0-50 --ec_level 3 --gpu 1 & # 3095830
# nohup python run_ml.py --reaction_level hard --protein_level 0-50 --ec_level 2 --gpu 1 & # 3095830


# Run also only on EC 3.1
# nohup python run_ml.py --label EC3-1 --reaction_level easy --protein_level 30 --ec_level 4 --data_dir /disk1/ariane/vscode/cec_degrader/experiment_24042025/data/ --gpu 0 --train_dataset CARE_training_datasets/protein_train_EC3.1.csv & # 3173463
# nohup python run_ml.py --label EC3 --reaction_level easy --protein_level 30 --ec_level 4 --data_dir /disk1/ariane/vscode/cec_degrader/experiment_24042025/data/ --gpu 0 --train_dataset CARE_training_datasets/protein_train_EC3.csv & # 3174000

# Run python run_ml.py --reaction_level easy --protein_level 30 --ec_level 4 --data_dir /disk1/ariane/vscode/cec_degrader/experiment_24042025/data/ --gpu 0