import cloudbutton
import os
import shutil


cloudbutton_dir = os.path.dirname(os.path.abspath(cloudbutton.__file__))
dst_backend_path = os.path.join(cloudbutton_dir, 'util/joblib')


if not os.path.isdir(os.path.join(cloudbutton_dir, 'util')):
    os.mkdir(os.path.join(cloudbutton_dir, 'util'))

if os.path.isdir(dst_backend_path):
    shutil.rmtree(dst_backend_path)
elif os.path.isfile(dst_backend_path):
    os.remove(dst_backend_path)


current_location = os.path.dirname(os.path.abspath(__file__))
src_backend_path = os.path.join(current_location, 'util/joblib')

shutil.copytree(src_backend_path, dst_backend_path)
