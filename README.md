# SessionRouteX

A session layer for multi-turn LLM routing.

In a chat, the right model for a message depends on the messages before it, and a model that already holds the chat in its KV cache answers the next message for less. SessionRouteX keeps this session state for every chat: the history (dialogue state) and which models hold the chat in their KV cache (serving state). Any router reads it through one interface, so multi-turn routers such as Chain-of-Route and single-turn routers plug in the same way.

**Status:** the code is being prepared for release and will be published in this repository.

## Contents

- **Toolkit**: the session layer between a chat application and its serving backends (vLLM or provider APIs), with one router interface.
- **[SessionRouteX Arena](arena/)**: compares routers on real chats in a static and a dynamic track.

## Paper

Jiaqi Xue, Mengxin Zheng, Heng Huang, Qian Lou. *Chain-of-Route: State-Aware LLM Routing for Multi-Turn Conversations.*
