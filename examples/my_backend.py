from sessionroutex import register_serving, Serving
from my_lab import my_client, PRICES    # your code

@register_serving
class MyBackend(Serving):
    name = "my-backend"
    models = ["gemma-26b-a4b", "gemma-31b"]

    def run(self, model, messages):
        reply = my_client.chat(model, messages)
        return reply.text, reply.usage    # usage: new and cached tokens

    def prices(self, model):
        return PRICES[model]              # $ per 1M tokens, cached and not
