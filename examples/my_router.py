from multiturnserve import register_router, Router
from multiturnserve.parts import CacheCost
from my_lab import MyQuality      # your code

@register_router
class MyRouter(Router):
    name = "my-router"
    quality = MyQuality()      # reads the history
    cost = CacheCost()         # reads the KV cache

    def decide(self, turn, state):
        q = self.quality(turn, state.history)
        c = self.cost(turn, state.kv_cache)
        return self.pick(q, c)

# after the reply, MultiTurnServe updates both states
