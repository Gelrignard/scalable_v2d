"""Rerun the local correction and audits using the saved real Tier 1 reference."""
from .retarget_bridge import WORKSPACE
from .book_trial import plan
from .robot_repair import repair
from .audit_book import audit

if __name__=='__main__':
    directory=WORKSPACE/'correction/outputs/tier1_book'
    plan(directory,135,176)
    repair(directory)
    audit(directory)
