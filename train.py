#!/usr/bin/env python
# coding: utf-8
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import neptune
from latent import *

print("Loading model...")
gpt_model = AutoModelForCausalLM.from_pretrained("gpt2-medium", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

print("Ready.")
model = LatentLM("gpt2-medium", model=gpt_model, tokenizer=tokenizer)

class Adapter(LatentModule):
    def __init__(self):
        super().__init__()
        
        self.lin1 = torch.nn.Linear(1024, 8)
        self.lin2 = torch.nn.Linear(8, 1024)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x

class DirectPrompt(torch.nn.Module):
    def forward(self, color):
        return model([["The boat is ", color, ". The color of the boat is"]], name='hard_color')

class SoftPrompt(torch.nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
    
    def forward(self, color):
        soft_color = model([["The boat is ", color, "."]], name='soft_color')[-2]
        soft_color = self.adapter(soft_color)
        return model([[soft_color, "The color of the boat is"]], name='soft_color')

colors = []
with open("working_colors.txt") as f:
    for color in f:
        colors.append(color.strip())
print(colors)

# test = colors[:30]
# batch_size = 2
# n = 0
# for i in range(0, len(test), batch_size):
#     batch = test[i:i+batch_size]
#     n += LatentModule.token_match(color_match(batch)).sum().item()
#     print(batch, direct(batch).discrete(), soft(batch).discrete())
#     print(n)


def experiment(loss_fct, lr=1e-4, epochs=10):
    run = neptune.init_run(
        project="lbeurerkellner/lat",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZDk4ZjE0ZS0wM2M1LTRkZmItODcxMC02OGYxOGFmZWM3OGYifQ==",
    )  # your credentials
    
    adapter = Adapter().to(model.device)
    soft = SoftPrompt(adapter).to(model.device)
    direct = DirectPrompt().to(model.device)

    def color_match(colors: List[str]):
        x = soft(colors)
        y = direct(colors)
        return x.layer(-1)[-1] == y.layer(-1)[-1]

    params = {"learning_rate": lr, "loss_fct": loss_fct, "epochs": epochs}
    run["parameters"] = params

    def reporter(epoch, train_acc, test_acc, loss):
        run["train/loss"].append(loss)
        run["train/accuracy"].append(train_acc)
        run["test/accuracy"].append(test_acc)
    
    train, test = colors[:30], colors[30:]
    train_acc, test_acc = adapter.fit(color_match, train, epochs=epochs, lr=1e-4, loss_fct=loss_fct, test=test, epoch_callback=reporter)
    
    run["train/accuracy"].log(train_acc)
    run["test/accuracy"].log(test_acc)

    run.stop()

if __name__ == "__main__":
    for lf in ["mse", "crossentropy", "cosine", "mse+cosine", "mse+crossentropy", "cosine+crossentropy", "cosine+crossentropy+mse"]:
        for lr in [1e-3, 1e-4, 1e-5]:
            experiment(lf, lr=lr, epochs=1000)




