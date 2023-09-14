from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import torch
from typing import Any, Union, List
from tqdm import tqdm

def leftpadstack(tensors, padding_value=0):
    max_len = max([t.shape[0] for t in tensors])
    padded_tensors = []
    mask = []
    
    if not torch.is_tensor(padding_value):
        padding_value = torch.tensor(padding_value, device=tensors[0].device).view(1,1)

    for t in tensors:
        padding = max_len - t.shape[0]
        padded_tensors += [torch.cat([padding_value] * (max_len - t.shape[0]) + [t], dim=0)]
        mask += [torch.cat([torch.zeros(max_len - t.shape[0], device=t.device), torch.ones(t.shape[0], device=t.device)], dim=0)]
    
    return torch.stack(padded_tensors, dim=0), torch.stack(mask, dim=0)

class LatentLM:
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model: AutoModelForCausalLM = model
        assert hasattr(self.model, 'forward'), "Expected 'transformers' model, but got {}".format(type(self.model))
        self.tokenizer = tokenizer

        self.input_id_embedding_module = "get_input_embeddings"

    @staticmethod
    def from_pretrained(model_name, **kwargs):
        model = kwargs.pop('model', None)
        tokenizer = kwargs.pop('tokenizer', None)

        return LatentLM(
            model_name,
            model or AutoModelForCausalLM.from_pretrained(model_name, **kwargs),
            tokenizer or AutoTokenizer.from_pretrained(model_name, **kwargs)
        )
    
    @property
    def device(self):
        return self.model.device

    def embed_input_ids(self, input_ids):
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()(input_ids)
        if hasattr(self.model, self.input_id_embedding_module):
            return getattr(self.model, self.input_id_embedding_module)(input_ids)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, self.input_id_embedding_module):
            return getattr(self.model.transformer, self.input_id_embedding_module)(input_ids)
        else:
            assert False, "Cannot embed input_ids for model '{}'. Please set the 'input_id_embedding_module' ({}) property to the name of the module that embeds input_ids.".format(self.model_name, self.input_id_embedding_module)

    def consolidate_prompt(self, prompt):
        if type(prompt) is list:
            r = []
            for s in prompt:
                if len(r) > 0 and type(s) is str and type(r[-1]) is str:
                    r[-1] += s
                else:
                    r += [s]
            return r
        else:
            return prompt

    def broadcast(self, prompt):
        shapes = list(set([len(s) if type(s) is list else 1 for s in prompt]))
        
        if len(shapes) == 1:
            if shapes[0] == 1:
                # all segments are singular
                return [[self.consolidate_prompt(prompt)]]
            else:
                # pair all segments index-wise
                return zip(*[prompt])
        elif len(shapes) == 2:
            assert shapes[0] == 1 or shapes[1] == 1, "Cannot broadcast prompt segments of shape {}".format(shapes)
            non_singular = shapes[0] if shapes[0] != 1 else shapes[1]
            pairings = []
            for i in range(non_singular):
                pairing = []
                for segment in prompt:
                    if type(segment) is list:
                        pairing += [segment[i]]
                    else:
                        pairing += [segment]
                pairings += [self.consolidate_prompt(pairing)]
            return pairings
        else:
            assert False, "Cannot broadcast prompt segments of shape {}".format(shapes)
        

    def embed(self, prompt: Union[List[Union[str, torch.Tensor]], str]) -> torch.Tensor:
        inputs_embeds = []
        text = ""
        offset_mapping = []
        input_ids = []
        first = True

        for segment in prompt:
            if type(segment) is str:
                segment_text = segment
                text += segment_text
                encoded_inputs = self.tokenizer(segment_text, return_tensors='pt', return_offsets_mapping=True, add_special_tokens=False)
                if first:
                    offset_mapping += torch.cat([torch.tensor([[len(text),len(text)]], dtype=torch.int64), encoded_inputs['offset_mapping'][0] + len(text)])
                    segment_ids = [self.tokenizer.bos_token_id] + encoded_inputs['input_ids'].flatten().tolist()
                else:
                    offset_mapping += encoded_inputs['offset_mapping'][0] + len(text)
                    segment_ids = encoded_inputs['input_ids'].flatten().tolist()
                input_ids += segment_ids
                segment = self.embed_input_ids(torch.tensor(segment_ids, dtype=torch.int64, device=encoded_inputs['input_ids'].device))
                inputs_embeds.append(segment)
            elif type(segment) is LatentTensor:
                latent: LatentTensor = segment
                latent_shp = latent.shape
                assert len(latent_shp) == 3, "Expected latent of shape (layers,seq_len,hidden_size), but got {}".format(latent_shp)
                assert latent_shp[0] == 1, "Expected latent of shape (1,seq_len,hidden_size), but got {}".format(latent_shp)
                seq_len = latent_shp[1]

                offset_begin = len(text)
                
                latent_repr = f"[{latent.names[0]}]"
                
                text += latent_repr
                input_ids += [latent] * seq_len
                
                offset_mapping += torch.tensor([[offset_begin, offset_begin + len(latent_repr)]] + [[0,0] for _ in range(seq_len - 1)], dtype=torch.int64)
                inputs_embeds += [latent.hidden_states[0]]
            else:
                raise ValueError("Expected prompt to be a string or a list of strings and tensors, but got {}".format(type(segment)))
                
            first = False

        inputs_embeds = torch.cat(inputs_embeds, dim=0)
        offset_mapping = torch.stack(offset_mapping, dim=0)
        attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long)

        return inputs_embeds, input_ids, text, offset_mapping, attention_mask

    def __call__(self, prompt, last_only=True, name=None, **kwargs):
        if type(prompt) is str:
            prompt = [prompt]
        
        batch_size = len(prompt)
        results = []
        for i in range(batch_size):
            for p in self.broadcast(prompt[i]):
                results += [self.embed(p)]
        batch_size = len(results)
        
        max_len = max([r[0].shape[0] for r in results])
        for i in range(batch_size):
            if results[i][0].shape[0] < max_len:
                # left-pad with zeros
                padded_inputs_embeds = torch.cat([torch.zeros(max_len - results[i][0].shape[0], results[i][0].shape[1], device=results[i][0].device), results[i][0]], dim=0)
                padded_input_ids = [0] * (max_len - results[i][0].shape[0]) + results[i][1]
                pad = torch.zeros(max_len - results[i][0].shape[0], device=results[i][0].device, dtype=torch.int64)
                padded_attention_mask = torch.cat([pad, results[i][4]], dim=0)
                # add (0,0) offsets for pad tokens
                offset_mapping = torch.cat([torch.zeros(max_len - results[i][0].shape[0], 2, device=results[i][0].device, dtype=torch.int32), results[i][3]], dim=0)

                results[i] = (padded_inputs_embeds, padded_input_ids, results[i][2], offset_mapping, padded_attention_mask)

        inputs_embeds = torch.stack([r[0] for r in results], dim=0)
        input_ids = [r[1] for r in results]
        text = [r[2] for r in results]
        offset_mapping = [r[3] for r in results]
        attention_mask = torch.stack([r[4] for r in results], dim=0)

        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            **kwargs
        }
        
        output = self.model(**model_inputs)
        
        return LatentTensor(output['hidden_states'], text, self, input_ids, offset_mapping, attention_mask, names=[name for _ in range(batch_size)])

def shape(v):
    if type(v) == list:
        return [len(v), *shape(v[0])]
    elif type(v) == tuple:
        return (len(v), *shape(v[0]))
    else:
        return v.shape

class LatentEqualityObjective:
    def __init__(self, *args):
        self.latents = args

        shape = self.latents[0].shape
        assert all([l.shape == shape for l in self.latents]), "Latents in equality objective have inconsistent shapes: {}. You can only assert equality between latents of the same shape.".format([l.shape for l in self.latents])

    def __str__(self):
        return "LatentEqualityObjective:\n" + "\n".join([" - " + str(l) for l in self.latents])

    def __repr__(self):
        return str(self)

class LatentTensor:
    """
    Represents an array of latents.
    """
    def __init__(self, hidden_states, prompts, model, input_ids, offset_mapping, attention_mask, names=None, layer_idx=None):
        self.hidden_states = hidden_states
        self.prompts = prompts
        self.model: LatentLM = model
        
        # if this is a latent from a specific layer_idx
        self.layer_idx = layer_idx
        
        self.input_ids = input_ids
        self.offset_mapping = offset_mapping
        self.attention_mask = attention_mask
        
        self.names = names or [None] * len(hidden_states)

    @property
    def shape(self):
        return list(shape(self.hidden_states))

    def __str__(self) -> str:
        title = "<" + self.model.model_name
        if self.layer_idx is not None:
            title += f".layer({self.layer_idx})"
        shp = self.shape[:-1]
        if self.layer_idx is not None:
            shp = shp[1:]

        title += ".LatentTensor{} <-> ".format(str(shp))

        prompts = [str([p + "[TOK]"])[1:-1] for n,p in zip(self.names, self.prompts)]
        prompts[1:] = [" " * len(title) + p for p in prompts[1:]]

        return title + "\n".join(prompts) + "\n"

    def __repr__(self):
        return str(self)
    
    def item(self, idx: Union[int, slice]):
        if type(idx) is int:
            if idx >= self.shape[1]:
                raise IndexError("Index {} out of range for LatentTensor of shape {}".format(idx, self.shape))
            idx = slice(idx, idx+1)

        attention_mask = self.attention_mask[idx]
        offset = (attention_mask == 0).sum(axis=-1).min().item()

        hidden_states = [hs[idx, offset:] for hs in self.hidden_states]
        prompt = self.prompts[idx]
        input_ids = self.input_ids[idx][offset:]
        offset_mapping = self.offset_mapping[idx][offset:]
        attention_mask = attention_mask[:,offset:]

        name = self.names[idx]

        return LatentTensor(hidden_states, prompt, self.model, input_ids, offset_mapping, attention_mask, names=name, layer_idx=self.layer_idx)

    def token(self, idx: Union[int, slice]):
        allows_empty = True
        
        if type(idx) is int:
            allows_empty = False
            if idx < 0:
                idx = slice(idx, idx + 1 if idx < -1 else None)
            else:
                idx = slice(idx, idx+1)
        
        # step is not supported
        if idx.step is not None:
            raise ValueError("LatentTensor token slicing does not support step size. Please use LatentTensor.item() instead.")

        attention_mask = self.attention_mask
        sample_offset = (attention_mask == 0).sum(axis=-1)
        seq_len = attention_mask.shape[1]

        def offset_pointer(ptr, offset, start=False):
            if ptr is None:
                if start:
                    return offset
                else:
                    return None
            elif ptr < 0:
                return ptr
            else:
                return ptr + offset

        subranges = [slice(offset_pointer(idx.start, o, start=True), offset_pointer(idx.stop, o)) for o in sample_offset]
        attention_masks = None
        
        # check for out-of-bounds
        for i, (mask, r) in enumerate(zip(attention_mask, subranges)):
            sample_len = seq_len - (mask == 0).sum().item()
            offset = (mask == 0).sum().item()
            try:
                assert allows_empty or len(range(sample_len)[r]) != 0
            except:
                raise IndexError("Index {} out of range for LatentTensor sample {} with input IDs {}".format(idx.start, i, self.input_ids[i][offset:]))

        # slice hidden states of each layer
        def sliced_layer(layer):
            nonlocal attention_masks
            result = []
            for l,r in zip(layer, subranges):
                result += [l[r]]
            result, attention_masks = leftpadstack(result, padding_value=torch.zeros(1, layer.shape[-1], device=layer.device))
            return result
        hidden_states = [sliced_layer(hs) for hs in self.hidden_states]

        # slice prompts
        offsets = self.offset_mapping
        prompts = []
        offset_mapping = []
        input_ids = []

        for r, prompt, offsets, ids in zip(subranges, self.prompts, self.offset_mapping, self.input_ids):
            pairs = offsets[r]

            if len(pairs) == 0:
                assert allows_empty, "Index {} out of range for LatentTensor sample {} with input IDs {}".format(idx.start, i, self.input_ids[i][offset:])
                prompt = ""
                pairs = torch.tensor([[0,0]], device=pairs.device)
            else:
                prompt = prompt[pairs[0][0]:pairs[-1][1]]
                shift = pairs[0][0]
                pairs = pairs - shift

            prompts += [prompt]
            offset_mapping += [pairs]
            input_ids += [ids[r]]
        
        return LatentTensor(hidden_states, prompts, self.model, input_ids, offset_mapping, attention_masks, names=self.names, layer_idx=self.layer_idx)

    def layer(self, idx: int):
        if idx >= self.shape[1]:
            raise IndexError("Index {} out of range for LatentTensor of shape {}".format(idx, self.shape))

        hidden_states = self.hidden_states[idx].unsqueeze(0)
        return LatentTensor(hidden_states, self.prompts, self.model, self.input_ids, self.offset_mapping, self.attention_mask, layer_idx=idx)

    def copy(self):
        return LatentTensor(self.hidden_states, self.prompts, self.model, self.input_ids, self.offset_mapping, self.attention_mask, names=self.names.copy(), layer_idx=self.layer_idx)
    
    def __getitem__(self, idx):
        return self.token(idx)
    
    def distribution(self, head="lm_head", softmax=True):
        """
        Returns the distribution over the vocabulary for the next token, 
        based on the current latent representation.

        :param head: The name of the head module to use for computing the distribution.
        :param softmax: Whether to apply log_softmax to the raw logits.
        """
        head_model = self.model.model._modules[head]

        bottom_right_hidden_state = self.hidden_states[-1][:,-1]
        logits = head_model(bottom_right_hidden_state)
        
        if softmax:
            return torch.log_softmax(logits, dim=-1)
        else:
            return logits
        
    def next(self, **kwargs) -> torch.IntTensor:
        """
        Returns the next token distribution.
        """
        temperature = kwargs.pop('temperature', 0.0)
        sample = kwargs.pop('sample', temperature != 0.0)
        if sample and temperature == 0.0:
            temperature = 1.0

        if sample:
            # temperature sampling
            logits = self.distribution().exp()
            logits = logits / temperature
            return torch.multinomial(logits, num_samples=1)
        else:
            return self.distribution().exp().argmax(-1)

    def complete(self, sample=False):
        """
        Calls the model and returns the current sequence appended with the 
        discretized next token as new LatentTensor.
        """
        # TODO: add key-value caching here
        return self.model(self.discrete(sample=sample))
    
    def discrete(self, **kwargs) -> str:
        """
        Discritizes the current latent representation by sampling a next token from it
        and returns the full text of the resulting sequence.
        """
        input_ids = self.input_ids.copy()
        updated_input_ids = []
        offsets = [(mask == 0).sum().item() for mask in self.attention_mask]
        
        for ids, next in zip(input_ids, self.next(**kwargs)):
            updated_input_ids += [[*ids, next.item()]]

        return [decode(uids[o:], self.model.tokenizer) for uids,o in zip(updated_input_ids, offsets)]

    def generate(self, max_tokens=32, sample=False):
        assert self.layer_idx is not None, "Can only .generate() from a specific layer. Please first select a specific .layer(<N>) to compute the token distribution for."

        x = self
        for _ in range(max_tokens):
            x = x.complete(sample=sample)
            x = x.layer(self.layer_idx)
        
        return x

# class Latent:
#     """
#     Represents a latent space of activations from a model.
    
#     Typical shape is (batch_size, sequence_length, hidden_size), when
#     coming straight from the model.
#     """

#     def __init__(self, activations, inputs, prompt, model, name=None):
#         self.activations = activations
#         self.inputs = inputs
#         self.prompt = prompt
#         self.name = name or "unnamed Latent"
#         self.model: LatentLM = model

#         if not torch.is_tensor(activations):
#             try:
#                 self.activations = torch.stack(activations)
#             except Exception as e:
#                 warnings.warn("Latent activations have inconsistent shapes and could not be stacked. Advanced latent slicing thus has to be done manually (loops).")

#     def copy(self):
#         return Latent(self.activations, self.inputs, self.prompt, self.model, name=self.name)

#     def __str__(self, full=False):
#         if full:
#             return f"""
# Prompt: {str([self.prompt])[1:-1]}
# Shape: {self.shape}
# Inputs: {self.inputs}
#             """
        
#         shp = shape(self.activations)[:-1]
#         if len(shp) != 0:
#             shp = f"{list(shp)}"
#         else:
#             shp = ""
        
#         if self.name != "unnamed Latent":
#             if shp == "":
#                 shp = f"[{self.name}]"
#             else:
#                 shp = f"[{self.name}]" + shp
        
#         s_repr = "lat.Latent{} <-> {}".format(shp, str([self.prompt + "[TOK]"])[1:-1])
        
#         return s_repr

#     def __repr__(self):
#         return str(self)

#     def __getitem__(self, idx):
#         prompt = "<substring of '" + self.prompt + "'>"
        
#         input_ids = self.inputs['input_ids']
#         attention_mask = self.inputs['attention_mask']
#         offsets = self.inputs['offset_mapping']
#         prompt = self.prompt

#         if len(self.shape) == 3:
#             if type(idx) is int:
#                 pass
#             elif type(idx) is tuple and len(idx) > 1:
#                 seq_idx = idx[1]

#                 if type(seq_idx) is int:
#                     input_ids = input_ids[seq_idx]
#                     attention_mask = attention_mask[seq_idx]
#                     offsets = offsets[seq_idx]
#                     shift = offsets[0]
#                     prompt = prompt[offsets[0]:offsets[1]]
#                     offsets = offsets - shift
#                 elif type(seq_idx) is slice:
#                     input_ids = input_ids[seq_idx]
#                     attention_mask = attention_mask[seq_idx]
#                     offsets = offsets[seq_idx]
#                     if len(offsets) == 0:
#                         prompt = ""
#                     else:
#                         shift = offsets[0][0]
#                         prompt = prompt[offsets[0][0]:offsets[-1][1]]
#                         offsets = offsets - shift
#         elif len(self.shape) == 2:
#             seq_idx = idx

#             if type(seq_idx) is int:
#                 input_ids = input_ids[seq_idx]
#                 attention_mask = attention_mask[seq_idx]
#                 offsets = offsets[seq_idx]
#                 shift = offsets[0]
#                 prompt = prompt[offsets[0]:offsets[1]]
#                 offsets = offsets - shift
#             elif type(seq_idx) is slice:
#                 input_ids = input_ids[seq_idx]
#                 attention_mask = attention_mask[seq_idx]
#                 offsets = offsets[seq_idx]
#                 shift = offsets[0][0]
#                 prompt = prompt[offsets[0][0]:offsets[-1][1]]
#                 offsets = offsets - shift
#         else:
#             assert False, "Cannot slice latents of shape {}. Please provide latents either as (layers, sequence_length, hidden_size) or (sequence_length, hidden_size).".format(self.shape)

#         inputs = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'offset_mapping': offsets
#         }

#         return Latent(self.activations[idx], inputs, prompt, self.model, name=self.name)

#     @property
#     def input_ids(self):
#         return self.inputs['input_ids']

#     def __len__(self):
#         return len(self.activations)
    
#     def __iter__(self):
#         return iter(self.activations)
    
#     @property
#     def last(self):
#         return self[-1]
    
#     @property
#     def seqlen(self):
#         if len(self.shape) == 3:
#             return self.shape[1]
#         elif len(self.shape) == 2:
#             return self.shape[0]
#         else:
#             return None

#     def token_distribution(self, head="lm_head", softmax=True):
#         """
#         Returns a distribution over the vocabulary for the given token index.

#         :param head: The name of the head module to use for computing the distribution.
#         :param softmax: Whether to apply log_softmax to the raw logits.
#         """
#         head_model = self.model.model._modules[head]
        
#         if len(self.shape) == 3:
#             bottom_right_hidden_state = self.activations[-1, -1]
#         elif len(self.shape) == 2:
#             bottom_right_hidden_state = self.activations[-1]
#         elif len(self.shape) == 1:
#             bottom_right_hidden_state = self.activations
#         else:
#             assert False, "Cannot compute token distribution for latents of shape {}. Please provide latents either as (layers, sequence_length, hidden_size) or (sequence_length, hidden_size).".format(self.shape)

#         logits = head_model(bottom_right_hidden_state)
        
#         if softmax:
#             return torch.log_softmax(logits, dim=-1)
#         else:
#             return logits
        
#     def next(self, **kwargs) -> int:
#         """
#         Returns the next token distribution.
#         """
#         temperature = kwargs.pop('temperature', 0.0)
#         sample = kwargs.pop('sample', temperature != 0.0)
#         if sample and temperature == 0.0:
#             temperature = 1.0

#         if sample:
#             # temperature sampling
#             logits = self.token_distribution().exp()
#             logits = logits / temperature
#             return torch.multinomial(logits, num_samples=1).item()
#         else:
#             return self.token_distribution().exp().argmax(-1).item()

#     def complete(self, sample=False):
#         """
#         Returns the next token distribution as a string.
#         """
#         # TODO: add key-value caching here
#         return self.model(self.discrete(sample=sample))
    
#     def discrete(self, **kwargs) -> str:
#         """
#         Discritizes the current latent representation by sampling a next token from it
#         and returns the full text of the resulting sequence.
#         """
#         input_ids = self.input_ids
#         if hasattr(input_ids, 'tolist'): 
#             input_ids = input_ids.tolist()
#         if type(input_ids) is not list: 
#             input_ids = [input_ids]
#         updated_input_ids = input_ids + [self.next(**kwargs)]
        
#         return decode(updated_input_ids, self.model.tokenizer)

#     def generate(self, max_tokens=32, sample=False):
#         x = self
#         for _ in range(max_tokens):
#             x = x.complete(sample=sample)
#         return x

#     @property
#     def shape(self):
#         return shape(self.activations)
    
#     def __eq__(self, other):
#         if type(other) is not Latent:
#             return False
#         return LatentEqualityObjective(self, other)
    
def decode(input_ids, tokenizer, hide_eos=True):
    if len(input_ids) == 0:
        return ""

    if type(input_ids[0]) is LatentTensor:
        latent = input_ids[0]
        return f"[{latent.names[0]}]" + decode(input_ids[1:], tokenizer)

    i = 0
    while i + 1 < len(input_ids) and type(input_ids[i+1]) is not LatentTensor:
        i += 1
    
    s = tokenizer.decode(input_ids[:i+1]) + decode(input_ids[i+1:], tokenizer)

    if hide_eos and s.endswith(tokenizer.eos_token):
        s = s[:-len(tokenizer.eos_token)]
    if hide_eos and s.startswith(tokenizer.bos_token):
        s = s[len(tokenizer.bos_token):]
    
    return s

class LatentModule(torch.nn.Module):
    def __init__(self):
        self.name = type(self).__name__
        super().__init__()

    def __call__(self, latent: LatentTensor, *args, **kwargs) -> LatentTensor:
        x = self.forward(latent.hidden_states[-1], *args, **kwargs)
        names = [f"{self.name}({n})" for n in latent.names]
        return LatentTensor(x, latent.prompts, latent.model, latent.input_ids, latent.offset_mapping, latent.attention_mask, names=names, layer_idx=latent.layer_idx)
    
    @staticmethod
    def loss(objective: LatentEqualityObjective, loss_fct="cosine"):
        """
        Fits the latent module such that the activations of l1 are transformed into the activations of l2.
        """
        assert len(objective.latents) == 2, "Expected exactly 2 latents, but got {}".format(len(objective.latents))
        
        device = objective.latents[0].activations.device
        loss = torch.tensor(0.0, device=device)

        if "mse" in loss_fct.lower():
            loss += torch.nn.MSELoss()(objective.latents[0].activations, objective.latents[1].activations)
        
        if "cosine" in loss_fct.lower():
            # print shapes
            loss += 1 - torch.nn.functional.cosine_similarity(objective.latents[0].activations.unsqueeze(0), objective.latents[1].activations.unsqueeze(0)).mean()

        if "crossentropy" in loss_fct.lower():
            target = torch.tensor(objective.latents[1].next(), device=device).unsqueeze(0)
            loss += torch.nn.CrossEntropyLoss()(objective.latents[0].distribution().unsqueeze(0), target)

        return loss

    def token_match(objective: LatentEqualityObjective, **kwargs):
        assert len(objective.latents) == 2, "Expected exactly 2 latents, but got {}".format(len(objective.latents))
        return objective.latents[0].next(**kwargs) == objective.latents[1].next(**kwargs)

    def fit(self, objective_fct, samples, loss_fct="cosine", epochs=1, lr=0.01, test=None, **kwargs):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, **kwargs)
        
        pbar = tqdm(range(epochs*len(samples)), desc="Epoch", position=0, leave=True)
        moving_loss = 0.0

        for e in range(epochs):
            for s in samples:
                optimizer.zero_grad()
                loss = LatentModule.loss(objective_fct(s), loss_fct=loss_fct)
                loss.backward()
                optimizer.step()

                moving_loss = 0.9 * moving_loss + 0.1 * loss.item()
                # check float for nan
                if moving_loss != moving_loss:
                    moving_loss = -1.0
                
                pbar.set_description(f"Epoch {e+1}/{epochs}, loss={moving_loss:.4f}")
                pbar.update(1)
            
            if test is not None:
                num_matches_test = 0
                for s in test:
                    num_matches_test += LatentModule.token_match(objective_fct(s))
                test_accuracy = 100 * num_matches_test / len(test)

                num_matches_train = 0
                for s in samples:
                    num_matches_train += LatentModule.token_match(objective_fct(s))
                train_accuracy = 100 * num_matches_train / len(samples)
                print("Loss: {:.4f}".format(moving_loss), "Train accuracy: {:.2f}%".format(train_accuracy), "Test accuracy: {:.2f}%".format(test_accuracy))

        pbar.close()