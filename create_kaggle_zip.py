import zipfile
import os
import sys

def make_zip(zip_filename='kaggle_training_test.zip'):
    files_to_zip = [
        'bivad_env.py',
        'reward.py',
        'train_centralized.py',
        'train_independent.py',
        'bilateral_extractor.py',
        'train_bilateral.py',
        'run_mpc_suction.py',
        'evaluate_all.py',
        'post_training_evaluation.py',
        'expert_data_pi.npz',
        'requirements.txt',
    ]

    dirs_to_zip = [
        'simulator',
        'controllers',
    ]

    print(f"Creating {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in files_to_zip:
            if os.path.exists(f):
                zipf.write(f, arcname=f)
                sz = os.path.getsize(f)
                print(f"  Added file: {f:25s} ({sz:,} bytes)")
            else:
                print(f"  WARNING: {f} not found!")

        for d in dirs_to_zip:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    if '__pycache__' in root:
                        continue
                    for file in files:
                        if file.endswith(('.pyc', '.pyo')):
                            continue
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, '.')
                        zipf.write(file_path, arcname=arcname)
                        print(f"  Added: {arcname}")

    total_size = os.path.getsize(zip_filename)
    print(f"\n[SUCCESS] Created {zip_filename} ({total_size:,} bytes / {total_size/(1024*1024):.2f} MB)")

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'kaggle_training_test.zip'
    make_zip(target)
