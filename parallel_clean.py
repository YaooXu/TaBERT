import os
import sys
import re
import pickle
import unicodedata
import os.path as osp
from tqdm import tqdm

import ujson
import pyarrow as pa
from pyarrow import feather
import multiprocessing as mp
from multiprocessing import Pool
from collections import Counter

CAP_TAG = "<caption>"
HEADER_TAG = "<header>"
ROW_TAG = "<row>"

MISSING_CAP_TAG = '[TAB]'
MISSING_CELL_TAG = "[CELL]"
MISSING_HEADER_TAG = "[HEAD]"

MAX_ROW_LEN = 100
MAX_COL_LEN = 100
MAX_WORD_LEN = 128



def clean_wiki_template(text):
    if re.match(r'^{{.*}}$', text):
        text = text[2:-2].split('|')[-1]  
    else:
        text = re.sub(r'{{.*}}', '', text)

    return text


def sanitize_text(text, entity="cell", replace_missing=True):
    """
    Clean up text in a table to ensure that the text doesn't accidentally
    contain one of the other special table tokens / tags.

    :param text: raw string for one cell in the table
    :return: the cell string after sanitizing
    """
    rval = re.sub(r"\|+", " ", text).strip()
    rval = re.sub(r'\s+', ' ', rval).strip()
    if rval and rval.split()[0] in ['td', 'th', 'TD', 'TH']:
        rval = ' '.join(rval.split()[1:])

    rval = rval.replace(CAP_TAG, "")
    rval = rval.replace(HEADER_TAG, "")
    rval = rval.replace(ROW_TAG, "")

    rval = rval.replace(MISSING_CAP_TAG , "")
    rval = rval.replace(MISSING_CELL_TAG, "")
    rval = rval.replace(MISSING_HEADER_TAG, "")
    rval =  ' '.join(rval.strip().split()[:MAX_WORD_LEN])

    # if (rval == "" or rval.lower() == "<missing>" or rval.lower() == "missing") and replace_missing:
    #     if entity == "cell":
    #         rval = MISSING_CELL_TAG
    #     elif entity == "header":
    #         rval = MISSING_HEADER_TAG
    #     else:
    #         rval = MISSING_CAP_TAG

    return rval




def clean_cell_value(cell_val):
    if isinstance(cell_val, list):
        val = ' '.join(cell_val)
    else:
        val = cell_val
    val = unicodedata.normalize('NFKD', val)
    val = val.encode('ascii', errors='ignore')
    val = str(val, encoding='ascii')
    val = clean_wiki_template(val)
    val = re.sub(r'\s+', ' ', val).strip()
    val = sanitize_text(val)
    return val






def read_json(name, all_graphs):
    with open(name, 'r') as f:
        for line in tqdm(f, desc='Loading Tables...', unit=' entries', file=sys.stdout):
            smpl = ujson.loads(line)
            result = json2string(smpl)
            if result is not None:
                all_graphs.append(result)

    return all_graphs



def json2string(exm):
    lower_cells, lower_heads = [], []
    # parser the table
    try:
        tb = exm['table']
    except:
        tb = exm
    cap = '' if tb['caption'] is None else tb['caption']
    # for now only keep some (rows, columns, words) in the arrow files.
    header = [h['name'] for h in tb['header']][:MAX_COL_LEN]
    data = tb['data']

    while len(data):
        if (header[0] and isinstance(data[0][0], list) and (header[0] in ' '.join(data[0][0]))):
            data = data[1:]
        else:
            break
    if not len(data):
        return None

    data = [row[:MAX_COL_LEN] for row in data[:MAX_ROW_LEN]]


    # sanitize the text
    cap = sanitize_text(cap, entity = 'cap')
    header = [sanitize_text(h, entity='header') for h in header]
    lower_heads.extend([h.lower() for h in header])
    
    cells = [list(map(clean_cell_value, row)) for row in data]
    for i, row in enumerate(cells):
        cells[i] = [cell.lower() for cell in row]
    
    # cells = [' | '.join(row) for row in cells]
    # text = ' '.join([CAP_TAG, cap, HEADER_TAG, header])
    # cell_text = ' '.join([ROW_TAG + ' '.format(i) + row for i, row in enumerate(cells)])
    # text = ' '.join([text, cell_text])
    
    return {
        'table': {
            "header": lower_heads,
            "rows": cells
        }
    }



def preprocess():

    input_dir = osp.join('./data/pretrain/', 'chunks')
    # input_dir = osp.join('./data/pretrain/', 'debug_chunks')
    output_dir = osp.join('./data/pretrain/', 'arrow')

    files = []
    for dirpath, _, filenames in os.walk(input_dir):
        for f in filenames:
            files.append(osp.abspath(osp.join(dirpath, f)))

    # process the data in parallel
    mg = mp.Manager()
    all_graphs = mg.list()
    pool = Pool(processes=len(files))
    for name in files:
        pool.apply_async(read_json, args=(name, all_graphs))
    pool.close()
    pool.join()
    
    serialize_text_to_arrow(all_graphs, folder=output_dir)

    # print('Counting for ELECTRA pretraining....')
    # heads_counter = Counter(all_lower_heads)
    # cells_counter = Counter(all_lower_cells)
    # print('Storing...')
    # with open(osp.join(output_dir, 'heads_counter.pkl'), 'wb') as f:
    #     pickle.dump(heads_counter, f)

    # with open(osp.join(output_dir, 'cells_counter.pkl'), 'wb') as f:
    #     pickle.dump(cells_counter, f)




def serialize_text_to_arrow(all_text, folder, split=None):
    print("Total lines: ", len(all_text))
    # Serialize to arrow format
    print("Starting serializing data")

    from datasets import Dataset
    dataset = Dataset.from_list(all_text)
    dataset.to_parquet("./data/pretrain/data.pq")


if __name__ == "__main__":
    preprocess()
