import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class HOT2020Dataset(BaseDataset):
    """ HOT2020 dataset
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.hot2020_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'hot2020', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        import os
        srcRootDir = self.base_path # '/data/lizf/HOT/dataset/test/test_HSI/'
        vdsLis = os.listdir(srcRootDir)
        vdsLis.sort()
        def getNum(vds):
            path = os.path.join(srcRootDir, vds, 'HSI')
            imgLis = os.listdir(path)
            cnt = 0
            for img in imgLis:
                if img.find('.png') != -1:
                    cnt += 1
            return cnt
        sequence_info_list = []
        for vds in vdsLis:
            ssDic = {"name": "forest", "path": "forest","startFrame": 1, "endFrame": 530, "nz": 4, "ext": "png", "anno_path": "forest/groundtruth_rect.txt", "object_class": "person"}
            num = getNum(vds)
            ssDic["name"] = vds
            ssDic['path'] = os.path.join(vds, 'HSI')
            ssDic['endFrame'] = num
            ssDic['anno_path'] = os.path.join(vds, 'groundtruth_rect.txt')
            sequence_info_list.append(ssDic)
    
        return sequence_info_list
