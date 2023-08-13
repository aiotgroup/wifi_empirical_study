import os
from data_process.util import load_mat

if __name__ == '__main__':
    """remote server"""
    datasource_paths = {
    }

    for datasource_name, datasource_path in datasource_paths.items():
        print(datasource_name.center(100, "="))
        data_filenames = os.listdir(datasource_path)
        # 只看第一层的mat文件
        for data_filename in data_filenames:
            if os.path.isdir(os.path.join(datasource_path, data_filename)) or data_filename.endswith(".mat"):
                continue
            mat_file = load_mat(os.path.join(datasource_path, data_filename))
            print(data_filename.center(100, "="))
            for key, value in mat_file.items():
                if key.startswith("__") or key.endswith("__"):
                    continue
                print(key, value.shape)

    # datasource_path = os.path.join("")
    # mat_file = scio.loadmat(os.path.join(datasource_path, 'train_dataset_none_0.25_0.10.mat'))
    # import pandas
    # csv_file = pandas.DataFrame(mat_file['data'][0])
    # csv_file.to_csv("./example.csv", header=False, index=False)