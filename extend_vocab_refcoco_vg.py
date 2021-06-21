from refer import REFER 
import os

def get_refer_classes():
    refer = REFER(dataset='refcoco', data_root='./data', splitBy='google')
    
    lastIdx = 1
    for key, value in refer.Cats.items():
        lastIdx = max(lastIdx, int(key))
    list_classes = ['None' for i in range(lastIdx+1)]
    for key, value in refer.Cats.items():
        list_classes[int(key)] = value
    return list_classes


if __name__ == "__main__":
    refcoco_classes = get_refer_classes()

    data_path_vg = "/projectnb/statnlp/gik/py-bottom-up-attention/demo/data/genome/1600-400-20"
    final_vocab_file = open('extended_objects_vocab_vg_recoco.txt', 'w+')

    with open(os.path.join(data_path_vg, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            final_vocab_file.write(object)

    for object in refcoco_classes:
        final_vocab_file.write(object+"\n")
    
    final_vocab_file.close()
