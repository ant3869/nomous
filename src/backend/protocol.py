# Title: UI Protocol helpers
# Path: backend/protocol.py
# Purpose: Small helpers that format messages the UI understands.

def msg_status(v: str, detail: str|None=None): return {"type":"status","value":v,"detail":detail}
def msg_event(message: str): return {"type":"event","message":message}
def msg_token(n: int): return {"type":"token","count":int(n)}
def msg_speak(text: str): return {"type":"speak","text":text}
def msg_image(dataUrl: str): return {"type":"image","dataUrl":dataUrl}
def msg_metrics(payload: dict): return {"type":"metrics","payload":payload}
def msg_memory(nodes, edges): return {"type":"memory","nodes":nodes,"edges":edges}
def msg_entities(entities: list): return {"type":"entities","entities":entities}
def msg_timeline(events: list): return {"type":"timeline","events":events}
def msg_search_results(results: list): return {"type":"search_results","results":results}
def msg_pong(): return {"type":"pong"}
