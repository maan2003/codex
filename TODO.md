context: working of new codex-acp (it is new code, so no backwards compat needed)

- [x] use conversation manager instead of managing conversation. because session id are constructed on agent side, we can just validate all passed sessionids into conversationids and fail if they are not valid uuids.
