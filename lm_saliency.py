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

plt.rcParams['figure.figsize'] = [10, 10]

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

    if foil is not None and correct != foil:
        (A.logits[-1][correct]-A.logits[-1][foil]).backward()
    else:
        (A.logits[-1][correct]).backward()
    handle.remove()
    hook.remove()

    return np.array(embeddings_gradients).squeeze(), np.array(embeddings_list).squeeze()


def integrated_gradients(
    inp, 
    target_label_index,
    predictions_and_gradients,
    baseline,
    steps=50):
  """Computes integrated gradients for a given network and prediction label.

  Integrated gradients is a technique for attributing a deep network's
  prediction to its input features. It was introduced by:
  https://arxiv.org/abs/1703.01365

  In addition to the integrated gradients tensor, the method also
  returns some additional debugging information for sanity checking
  the computation. See sanity_check_integrated_gradients for how this
  information is used.
  
  This method only applies to classification networks, i.e., networks 
  that predict a probability distribution across two or more class labels.
  
  Access to the specific network is provided to the method via a
  'predictions_and_gradients' function provided as argument to this method.
  The function takes a batch of inputs and a label, and returns the
  predicted probabilities of the label for the provided inputs, along with
  gradients of the prediction with respect to the input. Such a function
  should be easy to create in most deep learning frameworks.
  
  Args:
    inp: The specific input for which integrated gradients must be computed.
    target_label_index: Index of the target class for which integrated gradients
      must be computed.
    predictions_and_gradients: This is a function that provides access to the
      network's predictions and gradients. It takes the following
      arguments:
      - inputs: A batch of tensors of the same same shape as 'inp'. The first
          dimension is the batch dimension, and rest of the dimensions coincide
          with that of 'inp'.
      - target_label_index: The index of the target class for which gradients
        must be obtained.
      and returns:
      - predictions: Predicted probability distribution across all classes
          for each input. It has shape <batch, num_classes> where 'batch' is the
          number of inputs and num_classes is the number of classes for the model.
      - gradients: Gradients of the prediction for the target class (denoted by
          target_label_index) with respect to the inputs. It has the same shape
          as 'inputs'.
    baseline: [optional] The baseline input used in the integrated
      gradients computation. If None (default), the all zero tensor with
      the same shape as the input (i.e., 0*input) is used as the baseline.
      The provided baseline and input must have the same shape. 
    steps: [optional] Number of intepolation steps between the baseline
      and the input used in the integrated gradients computation. These
      steps along determine the integral approximation error. By default,
      steps is set to 50.

  Returns:
    integrated_gradients: The integrated_gradients of the prediction for the
      provided prediction label to the input. It has the same shape as that of
      the input.
      
    The following output is meant to provide debug information for sanity
    checking the integrated gradients computation.
    See also: sanity_check_integrated_gradients

    prediction_trend: The predicted probability distribution across all classes
      for the various (scaled) inputs considered in computing integrated gradients.
      It has shape <steps, num_classes> where 'steps' is the number of integrated
      gradient steps and 'num_classes' is the number of target classes for the
      model.
  """  
  if baseline is None:
    baseline = 0*inp
  assert(baseline.shape == inp.shape)

  # Scale input and compute gradients.
  scaled_inputs = [baseline + (float(i)/steps)*(inp-baseline) for i in range(0, steps+1)]
  grads, predictions= predictions_and_gradients(scaled_inputs, target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>
  
  # Use trapezoidal rule to approximate the integral.
  # See Section 4 of the following paper for an accuracy comparison between
  # left, right, and trapezoidal IG approximations:
  # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
  # https://arxiv.org/abs/1908.06214
  grads = (grads[:-1] + grads[1:]) / 2.0
  avg_grads = np.average(grads, axis=0)
  integrated_gradients = (inp-baseline)*avg_grads  # shape: <inp.shape>
  return integrated_gradients, predictions

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
def erasure_scores(model, input_ids, input_mask, correct=None, foil=None, remove=False, normalize=False):
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
        base_score = (probs[correct]-probs[foil]).detach().cpu().numpy()
    else:
        base_score = (probs[correct]).detach().cpu().numpy()

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
            erased_score = (probs[correct]-probs[foil]).detach().cpu().numpy()
        else:
            erased_score = (probs[correct]).detach().cpu().numpy()
                    
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
