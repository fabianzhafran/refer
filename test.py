# Only for testing how refer works

from refer import REFER

if __name__ == '__main__':
    dataroot = "/projectnb/statnlp/gik/refer/data"
    refer = REFER(dataset='refcoco', data_root=dataroot, splitBy='google')
    ref_ids = refer.getRefIds(split="test")[:]
    print(ref_ids[:4])