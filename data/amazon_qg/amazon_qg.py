import nlp
import pandas as pd

import pdb
class AmazonQACsvConfig(nlp.BuilderConfig):
        """BuilderConfig for amazon qa CSV."""
        def __init__(self, **kwargs):
            """BuilderConfig for AmazonQA.
            Args:
              **kwargs: keyword arguments forwarded to super.
            """
            super(AmazonQACsvConfig, self).__init__(**kwargs)
            

_DESCRIPTION = """This dataset contains Question and Answer data from Amazon, 
                totaling around 1.4 million answered questions."""
_CITATION = """ Modeling ambiguity, subjectivity, and diverging viewpoints 
                in opinion question answering systems
                Mengting Wan, Julian McAuley
                International Conference on Data Mining (ICDM), 2016

                Addressing complex and subjective product-related queries with 
                customer reviews Julian McAuley, Alex Yang World Wide Web (WWW), 2016"""

_TRAIN_FILE = 'data/amazon_qg/data/qa_Beauty_train.csv'
_VAL_FILE = 'data/amazon_qg/data/qa_Beauty_val.csv'

class AmazonQACsv(nlp.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = AmazonQACsvConfig

    def _info(self):
        return nlp.DatasetInfo(
        description=_DESCRIPTION,
        features=nlp.Features(
            {
                "source_text": nlp.Value("string"),
                "target_text": nlp.Value("string"),
                "task": nlp.Value("string"),           
            }
        ),
        # No default supervised_keys (as we have to pass both question
        # and context as input).
        supervised_keys=None,
        homepage="http://jmcauley.ucsd.edu/data/amazon/qa/",
        citation=_CITATION,
    )

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            return [nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": files}),
            nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"filepath": files})]

    def process_e2e_qg(self, pair):
        if not isinstance(pair['answer'], str):
            pair['answer'] = 'defualt string'
        source_text = f"generate questions: {pair['answer'].strip()}"
        target_text = f" {{sep_token}} {pair['question']}"
        target_text = f"{target_text} {{sep_token}}"
        return {"source_text": source_text, "target_text": target_text, "task": "e2e_qg"}

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)
        for i in range(len(df)):
            question = df.iloc[i]['question']
            answer = df.loc[i]['answer']
            pair =  {
                "answer": answer,
                "question": question
                }
            yield i, self.process_e2e_qg(pair)


# data = nlp.load_dataset('data/amazon_qg/', data_files=_VAL_FILE, split=nlp.Split.VALIDATION)
# train_dataset = nlp.load_dataset('data/amazon_qg/', data_files=_TRAIN_FILE, split=nlp.Split.TRAIN)
# valid_dataset = nlp.load_dataset('data/amazon_qg/', data_files=_VAL_FILE, split=nlp.Split.VALIDATION)


