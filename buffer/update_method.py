from buffer.reservoir_update import Reservoir_update
from buffer.aser_update import ASER_update
from buffer.gss_greedy_update import GSSGreedyUpdate
from buffer.summarize_update import SummarizeUpdate
from buffer.camel_update import CamelUpdate
from buffer.streamprompt_update import StreampromptUpdate

update_methods = {
    'random': Reservoir_update,
    'gss': GSSGreedyUpdate,
    'aser': ASER_update,
    'summarize': SummarizeUpdate,
    'camel': CamelUpdate,
    'streamprompt': StreampromptUpdate
}