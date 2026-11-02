# app.py
import multiturnserve

turns = ["Summarize this paper", "Shorter, please", "Compare to RouteLLM", "Thanks!"]   # the user's messages

s = multiturnserve.open("session-1", gateway=":8080")
for msg in turns:
    reply = s.chat(msg)       # route, run, observe
    print(reply.turn, reply.model, f"{reply.cost:.2e}")
