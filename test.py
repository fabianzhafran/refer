# Only for testing how refer works

from refer import REFER

dataroot = "/projectnb/statnlp/gik/refer/data"

if __name__ == '__main__':
    refer = REFER(dataset='refcocog',, dataroot splitBy='google')
    ref_ids = refer.getRefIds(split="test")[:]
    print(ref_ids)