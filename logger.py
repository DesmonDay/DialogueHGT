import logging

log = logging.getLogger('DialogueGCN')
console = logging.StreamHandler()
log.addHandler(console)
formatter = logging.Formatter(
    fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s'
)
console.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.propagate = False