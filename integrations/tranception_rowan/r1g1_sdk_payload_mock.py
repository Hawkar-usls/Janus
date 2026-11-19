import rowan,inspect,contextlib
from unittest.mock import patch
from types import SimpleNamespace
fn=rowan.submit_protein_cofolding_workflow
mod=inspect.getmodule(fn)
captured={}
class Stop(Exception):pass
class Fake:
    def post(self,url,json):
        captured.update(path=url,body=json)
        raise Stop
@contextlib.contextmanager
def mock():yield Fake()
with patch.object(mod,"api_client",mock):
    try:fn(initial_protein_sequences=["ACDEFGHIKLMNPQRSTVWY"],model=rowan.CofoldingModel.BOLTZ_2,num_samples=5,use_msa_server=True,max_credits=10,is_draft=False)
    except Stop:pass
assert captured["path"]=="/workflow" and captured["body"]["max_credits"]==10 and captured["body"]["workflow_data"]["num_samples"]==5
print("R1G1_PINNED_SDK_POST_PAYLOAD=PASS_MOCK_ONLY_NO_NETWORK")
