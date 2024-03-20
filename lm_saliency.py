import argparse, json
import random
import torch
import numpy as np
from transformers import (
    WEIGHTS_NAME,
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,

)

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [3, 3]

config = GPT2Config.from_pretrained("gpt2")
VOCAB_SIZE = config.vocab_size


def model_preds(model, input_ids, input_mask, pos, tokenizer, foils=None, k=10, verbose=False):
    # Obtain model's top predictions for given input
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(model.device)
    softmax = torch.nn.Softmax(dim=0)
    A = model(input_ids[:, :pos], attention_mask=input_mask[:, :pos])
    probs = softmax(A.logits[0][pos-1])
    top_preds = probs.topk(k)
    if verbose:
        if foils:
            for foil in foils:
                print("Contrastive loss: ", A.logits[0][pos-1][input_ids[0, pos]] - A.logits[0][pos-1][foil])
                print(f"{np.round(probs[foil].item(), 3)}: {tokenizer.decode(foil)}")
        print("Top model predictions:")
        for p,i in zip(top_preds.values, top_preds.indices):
            print(f"{np.round(p.item(), 3)}: {tokenizer.decode(i)}")
    return top_preds.indices

# Adapted from AllenNLP Interpret and Han et al. 2020
def register_embedding_list_hook(model, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    embedding_layer = model.transformer.wte
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def register_embedding_gradient_hooks(model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu().numpy())
    embedding_layer = model.transformer.wte
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook

def saliency(model, input_ids, input_mask, batch=0, correct=None, foil=None):
    # Get model gradients and input embeddings
    torch.enable_grad()
    model.eval()
    embeddings_list = []
    handle = register_embedding_list_hook(model, embeddings_list)
    embeddings_gradients = []
    hook = register_embedding_gradient_hooks(model, embeddings_gradients)
    
    if correct is None:
        correct = input_ids[-1]
    input_ids = input_ids[:-1]
    input_mask = input_mask[:-1]
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(model.device)

    model.zero_grad()
    A = model(input_ids, attention_mask=input_mask)
    softmax = torch.nn.Softmax(dim=0)
    logits = A.logits[-1]
    if foil is not None and correct != foil:
        # print(f'logits[foil] = {logits[foil]}')
        # print(f'logits[correct] - logits[foil] = {logits[correct] - logits[foil]}')
        diff = logits[correct]-logits[foil]
        # if (logits[correct] < 0 and diff > 0) or (logits[correct] > 0 and diff < 0):
            # diff = diff * -1
        print(diff)
        diff.backward()
    else:
        (logits[correct]).backward()
    handle.remove()
    hook.remove()

    return np.array(embeddings_gradients).squeeze(), np.array(embeddings_list).squeeze()


def input_x_gradient(grads, embds, normalize=False):
    input_grad = np.sum(grads * embds, axis=-1).squeeze()

    if normalize:
        norm = np.linalg.norm(input_grad, ord=1)
        input_grad /= norm
        
    return input_grad

def l1_grad_norm(grads, normalize=False):
    l1_grad = np.linalg.norm(grads, ord=1, axis=-1).squeeze()

    if normalize:
        norm = np.linalg.norm(l1_grad, ord=1)
        l1_grad /= norm
    return l1_grad
def erasure_scores(model, input_ids, input_mask, correct=None, foil=None, remove=False, normalize=False, attr_fn = "probs"):
    model.eval()
    if correct is None:
        correct = input_ids[-1]
    input_ids = input_ids[:-1]
    input_mask = input_mask[:-1]
    input_ids = torch.unsqueeze(torch.tensor(input_ids, dtype=torch.long).to(model.device), 0)
    input_mask = torch.unsqueeze(torch.tensor(input_mask, dtype=torch.long).to(model.device), 0)
    
    A = model(input_ids, attention_mask=input_mask)
    softmax = torch.nn.Softmax(dim=0)
    logits = A.logits[0][-1]
    probs = softmax(logits)
    if foil is not None and correct != foil:
        if attr_fn == "probs":
            diff = probs[correct]-probs[foil]
        elif attr_fn == "logit":
            diff = logits[correct]-logits[foil]
        base_score = (diff).detach().cpu().numpy() #HIER
    else:
        if attr_fn == "probs":
            base_score = (probs[correct]).detach().cpu().numpy()  #HIER
        elif attr_fn == "logit":
            base_score = (logits[correct]).detach().cpu().numpy()  #HIER

    scores = np.zeros(len(input_ids[0]))
    for i in range(len(input_ids[0])):
        if remove:
            input_ids_i = torch.cat((input_ids[0][:i], input_ids[0][i+1:])).unsqueeze(0)
            input_mask_i = torch.cat((input_mask[0][:i], input_mask[0][i+1:])).unsqueeze(0)
        else:
            input_ids_i = torch.clone(input_ids)
            input_mask_i = torch.clone(input_mask)
            input_mask_i[0][i] = 0

        A = model(input_ids_i, attention_mask=input_mask_i)
        logits = A.logits[0][-1]
        probs = softmax(logits)
    if foil is not None and correct != foil:
        if attr_fn == "probs":
            diff = probs[correct]-probs[foil]
        elif attr_fn == "logit":
            diff = logits[correct]-logits[foil]
        erased_score = (diff).detach().cpu().numpy() #HIER
    else:
        if attr_fn == "probs":
            erased_score = (probs[correct]).detach().cpu().numpy()  #HIER
        elif attr_fn == "logit":
            erased_score = (logits[correct]).detach().cpu().numpy()  #HIER
                    
        scores[i] = base_score - erased_score # higher score = lower confidence in correct = more influential input
    if normalize:
        norm = np.linalg.norm(scores, ord=1)
        scores /= norm
    return scores

def visualize(attention, tokenizer, input_ids, gold=None, normalize=False, print_text=True, save_file=None, title=None, figsize=60, fontsize=36):
    tokens = [tokenizer.decode(i) for i in input_ids[0][:len(attention) + 1]]
    if gold is not None:
        for i, g in enumerate(gold):
            if g == 1:
                tokens[i] = "**" + tokens[i] + "**"

    # Normalize to [-1, 1]
    if normalize:
        a,b = min(attention), max(attention)
        x = 2/(b-a)
        y = 1-b*x
        attention = [g*x + y for g in attention]
    attention = np.array([list(map(float, attention))])

    fig, ax = plt.subplots(figsize=(figsize,figsize))
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(attention, cmap='seismic', norm=norm)

    if print_text:
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=fontsize)
    else:
        ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for (i, j), z in np.ndenumerate(attention):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=fontsize)


    ax.set_title("")
    fig.tight_layout()
    if title is not None:
        plt.title(title, fontsize=36)
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches = 'tight',
        pad_inches = 0)
        plt.close()
    else:
        plt.show()

def main():
    pass

if __name__ == "__main__":
    main()
