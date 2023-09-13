from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import torch
from typing import Any, Union, List
from tqdm import tqdm
class LatentLM:
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model: AutoModelForCausalLM = model
        assert hasattr(self.model, 'forward'), "Expected 'transformers' model, but got {}".format(type(self.model))
        self.tokenizer = tokenizer

        self.input_id_embedding_module = "wte"

    @staticmethod
    def from_pretrained(model_name, **kwargs):
        model = kwargs.pop('model', None)
        tokenizer = kwargs.pop('tokenizer', None)

        return LatentLM(
            model_name,
            model or AutoModelForCausalLM.from_pretrained(model_name, **kwargs),
            tokenizer or AutoTokenizer.from_pretrained(model_name, **kwargs)
        )
    
    def embed_input_ids(self, input_ids):
        if hasattr(self.model, self.input_id_embedding_module):
            return getattr(self.model, self.input_id_embedding_module)(input_ids)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, self.input_id_embedding_module):
            return getattr(self.model.transformer, self.input_id_embedding_module)(input_ids)
        else:
            assert False, "Cannot embed input_ids for model '{}'. Please set the 'input_id_embedding_module' property to the name of the module that embeds input_ids.".format(self.model_name)

    def embed(self, prompt: Union[List[Union[str, torch.Tensor]], str]) -> torch.Tensor:
        if type(prompt) is str:
            prompt = [prompt]

        inputs_embeds = []
        text = ""
        offset_mapping = []
        input_ids = []

        for segment in prompt:
            if type(segment) is str:
                segment_text = segment
                encoded_inputs = self.tokenizer(segment_text, return_tensors='pt', return_offsets_mapping=True)
                offset_mapping += encoded_inputs['offset_mapping'][0] + len(text)
                text += segment_text
                input_ids += encoded_inputs['input_ids'].flatten().tolist()
                segment = self.embed_input_ids(encoded_inputs['input_ids'])
                inputs_embeds.append(segment)
            elif type(segment) is Latent:
                latent = segment

                assert len(latent.shape) < 3, "Cannot embed latents of shape {}. Please provide latents either in shape (sequence_length, hidden_size) or (hidden_size).".format(latent.shape)

                offset_begin = len(text)
                if len(latent.shape) == 2:
                    latent = latent.copy()
                    latent.name += "(width={})".format(latent.shape[0])

                latent_repr = f"[{latent.name}]"
                
                text += latent_repr
                input_ids += [latent]
                
                offset_mapping += torch.tensor([[offset_begin, offset_begin + len(latent_repr)]])
                
                if len(latent.shape) == 2:
                    # multi-token latent
                    inputs_embeds += [latent.activations.unsqueeze(0)]
                else:
                    # single-token latent
                    inputs_embeds += [latent.activations.unsqueeze(0).unsqueeze(0)]
            else:
                raise ValueError("Expected prompt to be a string or a list of strings and tensors, but got {}".format(type(prompt)))

        return torch.cat(inputs_embeds, dim=1), input_ids, text, torch.stack(offset_mapping, dim=0)

    def __call__(self, *prompt, last_only=True, name=None, **kwargs):
        inputs_embeds, input_ids, text, offset_mapping = self.embed(prompt)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)

        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            **kwargs
        }
        
        output = self.model(**model_inputs)
        
        model_inputs["offset_mapping"] = offset_mapping
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask[0]

        # keep track of offset mapping
        if last_only:
            return Latent([s[0] for s in output['hidden_states']], model_inputs, text, self, name=name)[-1]
        return Latent([s[0] for s in output['hidden_states']], model_inputs, text, self, name=name)

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

class Latent:
    """
    Represents a latent space of activations from a model.
    
    Typical shape is (batch_size, sequence_length, hidden_size), when
    coming straight from the model.
    """

    def __init__(self, activations, inputs, prompt, model, name=None):
        self.activations = activations
        self.inputs = inputs
        self.prompt = prompt
        self.name = name or "unnamed Latent"
        self.model: LatentLM = model

        if not torch.is_tensor(activations):
            try:
                self.activations = torch.stack(activations)
            except Exception as e:
                warnings.warn("Latent activations have inconsistent shapes and could not be stacked. Advanced latent slicing thus has to be done manually (loops).")

    def copy(self):
        return Latent(self.activations, self.inputs, self.prompt, self.model, name=self.name)

    def __str__(self, full=False):
        if full:
            return f"""
Prompt: {str([self.prompt])[1:-1]}
Shape: {self.shape}
Inputs: {self.inputs}
            """
        
        shp = shape(self.activations)[:-1]
        if len(shp) != 0:
            shp = f"{list(shp)}"
        else:
            shp = ""
        
        if self.name != "unnamed Latent":
            if shp == "":
                shp = f"[{self.name}]"
            else:
                shp = f"[{self.name}]" + shp
        
        s_repr = "lat.Latent{} <-> {}".format(shp, str([self.prompt + "[TOK]"])[1:-1])
        
        return s_repr

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx):
        prompt = "<substring of '" + self.prompt + "'>"
        
        input_ids = self.inputs['input_ids']
        attention_mask = self.inputs['attention_mask']
        offsets = self.inputs['offset_mapping']
        prompt = self.prompt

        if len(self.shape) == 3:
            if type(idx) is int:
                pass
            elif type(idx) is tuple and len(idx) > 1:
                seq_idx = idx[1]

                if type(seq_idx) is int:
                    input_ids = input_ids[seq_idx]
                    attention_mask = attention_mask[seq_idx]
                    offsets = offsets[seq_idx]
                    shift = offsets[0]
                    prompt = prompt[offsets[0]:offsets[1]]
                    offsets = offsets - shift
                elif type(seq_idx) is slice:
                    input_ids = input_ids[seq_idx]
                    attention_mask = attention_mask[seq_idx]
                    offsets = offsets[seq_idx]
                    if len(offsets) == 0:
                        prompt = ""
                    else:
                        shift = offsets[0][0]
                        prompt = prompt[offsets[0][0]:offsets[-1][1]]
                        offsets = offsets - shift
        elif len(self.shape) == 2:
            seq_idx = idx

            if type(seq_idx) is int:
                input_ids = input_ids[seq_idx]
                attention_mask = attention_mask[seq_idx]
                offsets = offsets[seq_idx]
                shift = offsets[0]
                prompt = prompt[offsets[0]:offsets[1]]
                offsets = offsets - shift
            elif type(seq_idx) is slice:
                input_ids = input_ids[seq_idx]
                attention_mask = attention_mask[seq_idx]
                offsets = offsets[seq_idx]
                shift = offsets[0][0]
                prompt = prompt[offsets[0][0]:offsets[-1][1]]
                offsets = offsets - shift
        else:
            assert False, "Cannot slice latents of shape {}. Please provide latents either as (layers, sequence_length, hidden_size) or (sequence_length, hidden_size).".format(self.shape)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'offset_mapping': offsets
        }

        return Latent(self.activations[idx], inputs, prompt, self.model, name=self.name)

    @property
    def input_ids(self):
        return self.inputs['input_ids']

    def __len__(self):
        return len(self.activations)
    
    def __iter__(self):
        return iter(self.activations)
    
    @property
    def last(self):
        return self[-1]
    
    @property
    def seqlen(self):
        if len(self.shape) == 3:
            return self.shape[1]
        elif len(self.shape) == 2:
            return self.shape[0]
        else:
            return None

    def token_distribution(self, head="lm_head", softmax=True):
        """
        Returns a distribution over the vocabulary for the given token index.

        :param head: The name of the head module to use for computing the distribution.
        :param softmax: Whether to apply log_softmax to the raw logits.
        """
        head_model = self.model.model._modules[head]
        
        if len(self.shape) == 3:
            bottom_right_hidden_state = self.activations[-1, -1]
        elif len(self.shape) == 2:
            bottom_right_hidden_state = self.activations[-1]
        elif len(self.shape) == 1:
            bottom_right_hidden_state = self.activations
        else:
            assert False, "Cannot compute token distribution for latents of shape {}. Please provide latents either as (layers, sequence_length, hidden_size) or (sequence_length, hidden_size).".format(self.shape)

        logits = head_model(bottom_right_hidden_state)
        
        if softmax:
            return torch.log_softmax(logits, dim=-1)
        else:
            return logits
        
    def next(self, **kwargs) -> int:
        """
        Returns the next token distribution.
        """
        temperature = kwargs.pop('temperature', 0.0)
        sample = kwargs.pop('sample', temperature != 0.0)
        if sample and temperature == 0.0:
            temperature = 1.0

        if sample:
            # temperature sampling
            logits = self.token_distribution().exp()
            logits = logits / temperature
            return torch.multinomial(logits, num_samples=1).item()
        else:
            return self.token_distribution().exp().argmax(-1).item()

    def complete(self, sample=False):
        """
        Returns the next token distribution as a string.
        """
        # TODO: add key-value caching here
        return self.model(self.discrete(sample=sample))
    
    def discrete(self, **kwargs) -> str:
        """
        Discritizes the current latent representation by sampling a next token from it
        and returns the full text of the resulting sequence.
        """
        input_ids = self.input_ids
        if hasattr(input_ids, 'tolist'): 
            input_ids = input_ids.tolist()
        if type(input_ids) is not list: 
            input_ids = [input_ids]
        updated_input_ids = input_ids + [self.next(**kwargs)]
        
        return decode(updated_input_ids, self.model.tokenizer)

    def generate(self, max_tokens=32, sample=False):
        x = self
        for _ in range(max_tokens):
            x = x.complete(sample=sample)
        return x

    @property
    def shape(self):
        return shape(self.activations)
    
    def __eq__(self, other):
        if type(other) is not Latent:
            return False
        return LatentEqualityObjective(self, other)
    
def decode(input_ids, tokenizer):
    if len(input_ids) == 0:
        return ""

    if type(input_ids[0]) is Latent:
        latent = input_ids[0]
        return f"[{latent.name}]" + decode(input_ids[1:], tokenizer)

    i = 0
    while i + 1 < len(input_ids) and type(input_ids[i+1]) is not Latent:
        i += 1
    
    return tokenizer.decode(input_ids[:i+1]) + decode(input_ids[i+1:], tokenizer)

class LatentModule(torch.nn.Module):
    def __init__(self):
        self.name = type(self).__name__
        super().__init__()

    def __call__(self, latent: 'Latent', *args, **kwargs) -> 'Latent':
        activations = self.forward(latent.activations, *args, **kwargs)
        name = f"{self.name}({latent.name})"
        return Latent(activations, latent.inputs, latent.prompt, latent.model, name=name)
    
    @staticmethod
    def loss(objective: LatentEqualityObjective, loss_fct="cosine"):
        """
        Fits the latent module such that the activations of l1 are transformed into the activations of l2.
        """
        assert len(objective.latents) == 2, "Expected exactly 2 latents, but got {}".format(len(objective.latents))
        return torch.nn.MSELoss()(objective.latents[0].activations, objective.latents[1].activations)

    def token_match(objective: LatentEqualityObjective, **kwargs):
        assert len(objective.latents) == 2, "Expected exactly 2 latents, but got {}".format(len(objective.latents))
        return objective.latents[0].next(**kwargs) == objective.latents[1].next(**kwargs)

    def fit(self, objective_fct, samples, loss_fct="cosine", epochs=1, lr=0.01, **kwargs):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, **kwargs)
        
        pbar = tqdm(range(epochs*len(samples)), desc="Epoch", leave=False)
        moving_loss = 0.0

        for e in range(epochs):
            for s in samples:
                optimizer.zero_grad()
                loss = LatentModule.loss(objective_fct(s), loss_fct=loss_fct)
                loss.backward()
                optimizer.step()

                moving_loss = 0.9 * moving_loss + 0.1 * loss.item()
                
                pbar.set_description(f"Epoch {e+1}/{epochs}, loss={moving_loss:.4f}")
                pbar.update(1)
            
            num_matches = 0
            for s in samples:
                num_matches += LatentModule.token_match(objective_fct(s))

            print("Loss: {:.4f}".format(moving_loss), "Token match: {:.2f}%".format(100 * num_matches / len(samples)))

        pbar.close()