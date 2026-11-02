# app.py
import multiturnserve

chat = ["Summarize this paper", "Shorter, please", "Compare to RouteLLM", "Thanks!"]

s = multiturnserve.open("chat-1", gateway=":8080")
for msg in chat:
    reply = s.chat(msg)       # route, run, observe
    print(reply.turn, reply.model, reply.cost)
