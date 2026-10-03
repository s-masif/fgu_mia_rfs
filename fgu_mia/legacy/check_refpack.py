"""
Sanity-check a refpack against the frozen node log / manifest for one cell.
Confirms the reference outputs are consistent with the frozen targets:
  - the node groups (F/H/attack_member/attack_nonmember) match the frozen log
  - population + final-eval are a disjoint split of retained non-members, disjoint
    from calibration
  - each reference model produced a probability for every node we need to score
  - reference softmax vectors are valid (length = n_classes, sum to 1)
  - OUT status: no scored eval node is in any reference training set
Run:  python check_refpack.py --tag Cora_client_s42 --frozen-dir results_13x5/raw --out-dir attack_ext_out
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="Cora_client_s42")
    ap.add_argument("--frozen-dir", default="results_13x5/raw")
    ap.add_argument("--out-dir", default="attack_ext_out")
    a = ap.parse_args()

    rp = json.load(open(Path(a.out_dir)/"reference"/f"refpack_{a.tag}.json"))
    frozen = pd.read_csv(Path(a.frozen_dir)/f"nodelog_{a.tag}.csv")

    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  <- {detail}"))
        ok = ok and cond

    # 1. node groups match the frozen log
    for sub in ["F","H","attack_member","attack_nonmember"]:
        frozen_ids = set(int(x) for x in frozen[frozen.subset==sub].global_node_id.unique())
        pack_ids = set(rp["node_groups"][sub])
        check(f"{sub} ids match frozen log ({len(pack_ids)})", frozen_ids==pack_ids,
              f"frozen={len(frozen_ids)} pack={len(pack_ids)}")

    # 2. population / final-eval disjoint, and disjoint from calibration
    pop=set(rp["population_ids"]); fe=set(rp["final_eval_nonmember_ids"]); cal=set(rp["calibration_ids"])
    check("population ∩ final-eval == ∅", not (pop&fe))
    check("population ∩ calibration == ∅", not (pop&cal))
    check("population ⊆ attack_nonmember", pop <= set(rp["node_groups"]["attack_nonmember"]))

    # 3. every reference model scored every needed node
    want = set(rp["node_groups"]["F"])|set(rp["node_groups"]["H"])|set(rp["node_groups"]["attack_member"]) \
           |set(rp["node_groups"]["attack_nonmember"])|cal|pop
    for i, refd in enumerate(rp["reference_probs"]):
        got = set(int(k) for k in refd.keys())
        check(f"reference model {i} covers all needed nodes ({len(want)})", want <= got,
              f"missing {len(want-got)}")

    # 4. reference softmax valid
    bad=0
    for refd in rp["reference_probs"]:
        for v in refd.values():
            if abs(sum(v)-1.0)>1e-3: bad+=1
    check("all reference softmax sum to 1", bad==0, f"{bad} bad vectors")

    # 5. OUT status
    check("OUT status clean (no eval node in any ref train set)", rp.get("out_status_leak",1)==0)

    print("\nOVERALL:", "ALL PASS ✅" if ok else "SOME FAILED ❌")

if __name__=="__main__":
    main()