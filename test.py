# Only for testing how refer works

from refer import REFER

if __name__ == '__main__':
    refer = REFER(dataset='refcocog', splitBy='google')
    ref_ids = refer.getRefIds(split="test")[:]
    print(ref_ids)