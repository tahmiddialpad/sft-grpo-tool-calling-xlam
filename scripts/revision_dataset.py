"""Use the frozen xLAM revision template for every verl prompt."""
from pathlib import Path
from verl.utils.dataset.rl_dataset import RLHFDataset


class RevisionDataset(RLHFDataset):
    def __init__(self,data_files,tokenizer,config,**kwargs):
        tokenizer.chat_template=(Path(__file__).resolve().parents[1]/'metadata/revision_chat_template.jinja').read_text()
        super().__init__(data_files,tokenizer,config,**kwargs)
