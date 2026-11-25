"""R84H read-only retrospective DEP/RISK/PROFIT audit. No training, admission or promotion."""
import json,math,sys
from pathlib import Path
PINS={
"replay/fixed":["c3bb58e7ff89d529b982c9020d212bdadb97162669e5d5c15820600e092ba7e8","630b344c61cbd03890c1686737bf34d6557ea498b8e97c198bb5056e27ba9931","1b044fccd0d4bb4c6c6b0b3e6d4ac7e3c7d8f5222b32dd28a82007aba8d5d338"],
"replay/selective":["27fe73e5ea50175a9206cd7eb973128e8e0a914077c2c289ae207cde24f94412","d3aa71efadaa3e86ed42d55a71ed26461b894a3f26d44fc81e947f309a512673","fe071c521098db7a1925bd636542af6b4e7574ea0387e1785ef17d316287e6c3"],
"joint/fixed":["f76666f49573457cc499c47c8488dd0406954ac26d6bc80f4298bdc762d8bbfc","619f15c29303d5cf7c73636c4398e32a5ddbc8ec69988de86a08eeec90e5caf9","4d0c6ba9c54d7061ebd4791a04c3952aabbf229bc9647e3848806d64b090196b"],
"joint/selective":["50dbdbb5030eb43361adc51a34c9c106c3407205a35c2f6d6e5e079136c40905","8f6394649a480fe728af1028a37a74fb6c62e1e70fa4fd19db6354de97dd0a1f","6fe522e7b968b357879ceb0fb6ecfa3043c8e9ddd2dce946f9f7be83180190ca"]}
# exact prior run, same examination of 243 prompts across 3 seeds.
EXPECTED={"replay/fixed":(547,592,485),"replay/selective":(547,607,495),"joint/fixed":(511,563,450),"joint/selective":(511,552,444)}
def ck(ok,why):
    if not ok:raise ValueError(why)
def metrics(rows):
    before=sum(sum(x["before"]) for x in rows);after=sum(sum(x["after"]) for x in rows)
    total=sum(sum(x["n"]) for x in rows);retained=sum(sum(x["retained"]) for x in rows)
    seconds=sum(x["seconds"] for x in rows);updates=240*len(rows)
    ck(0<before<=total and 0<=retained<=min(before,after) and math.isfinite(seconds) and seconds>0,"METRICS")
    return {"dep_correct":after,"dep_accuracy":after/total,"n":total,"before":before,"risk_lost":before-retained,"risk_fraction":(before-retained)/before,"retained":retained,"net_gain":after-before,"profit_per_100_updates":100*(after-before)/updates,"profit_per_elapsed_training_second":(after-before)/seconds,"elapsed_training_seconds":seconds,"gradient_updates":updates}
def compare(f,s):
    return {"dep_up":s["dep_correct"]>f["dep_correct"],"risk_down":s["risk_fraction"]<f["risk_fraction"],"profit_update_up":s["profit_per_100_updates"]>f["profit_per_100_updates"],"profit_elapsed_up_observed":s["profit_per_elapsed_training_second"]>f["profit_per_elapsed_training_second"]}
def main(src,dest):
    obj=json.loads(src.read_text());ck(obj["schema"]=="r84g2.result.v1" and obj["production"] is False and obj["paired"] is True,"SOURCE")
    rows=obj["results"];ck(len(rows)==12,"ROWS")
    groups={}
    for x in rows:
        k=x["arm"]+"/"+x["mode"];i=x["seed"]-111
        ck(k in PINS and i in range(3) and x["model_sha256"]==PINS[k][i],"MODEL_PIN")
        ck(len(x["n"])==len(x["before"])==len(x["after"])==len(x["retained"])==len(x["lost_case_indices"])==15 and sum(x["n"])==243,"EXAM")
        for n,b,a,t,lost in zip(x["n"],x["before"],x["after"],x["retained"],x["lost_case_indices"]):
            ck(0<=t<=min(a,b)<=n and len(lost)==b-t and len(set(lost))==len(lost) and all(type(j)==int and 0<=j<n for j in lost),"CASE_COUNTS")
        groups.setdefault(k,{})[x["seed"]]=x
    ck(set(groups)==set(PINS) and all(set(q)=={111,112,113} for q in groups.values()),"GROUPS")
    results={}
    for k,q in groups.items():
        m=metrics(list(q.values()));ck((m["before"],m["dep_correct"],m["retained"])==EXPECTED[k],"LOG_MATCH")
        results[k]=m;print("R84H_METRIC",json.dumps({"line":k,**m}),flush=True)
    comparisons={}
    for arm in ("replay","joint"):
        k=arm+"/fixed";l=arm+"/selective";f,s=groups[k],groups[l]
        per_seed=[]
        for seed in sorted(f):
            x,y=f[seed],s[seed];ck(x["parent_sha256"]==y["parent_sha256"] and x["before"]==y["before"] and x["n"]==y["n"],"PAIR")
            per_seed.append({"seed":seed,**compare(metrics([x]),metrics([y]))})
        aggregate=compare(results[k],results[l])
        # Diagnostic, not a release gate. No promotion decision is emitted.
        obs={"aggregate":aggregate,"per_seed":per_seed,"all_seed_directions":all(x["dep_up"] and x["risk_down"] and x["profit_update_up"] for x in per_seed),"zero_forgetting":results[l]["risk_lost"]==0,"matched_total_compute":False,"timing_causal":False}
        comparisons[arm]=obs;print("R84H_COMPARISON",json.dumps({"arm":arm,**obs}),flush=True)
    dest.mkdir(parents=True,exist_ok=False);(dest/"result.json").write_text(json.dumps({"schema":"r84h.retrospective.v1","source_run":45688517804,"metrics":results,"comparisons":comparisons,"training":False,"production":False,"external_validation":False,"note":"Retrospective descriptive metrics on same 243 internal cases; not independent confirmation. DEP is repeated correct responses, not unique skills. RISK is observed forgetting. PROFIT per updates excludes extra probes; elapsed timing is cross-runner observational."},sort_keys=True,indent=2)+"\n")
if __name__=="__main__":main(Path(sys.argv[1]),Path(sys.argv[2]))
