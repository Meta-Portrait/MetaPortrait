import importlib
from basicsr.utils import scandir
from os import path as osp

# automatically scan and import arch modules for registry
# scan all the files that end with '_arch.py' under the archs folder
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
new_list  = []
for f in arch_filenames:
    if 'GPEN' not in f and 'gpen' not in f:
        new_list.append(f)
arch_filenames = new_list
# print("import" + str(arch_filenames))
_arch_modules = [importlib.import_module(f'Experimental_root.archs.{file_name}') for file_name in arch_filenames]
