# tensor_prep.py
import torch

def prepare_tensors(
    dev_normal_idx, dev_abnormal_idx, test_abnormal_idx,
    W_dev, X_s_dev, W_test, X_s_test
):
    # device 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 인덱스 텐서 변환
    selected_indices      = [torch.tensor(dev_normal_idx, device=device)]
    selected_indices_test = [torch.tensor(dev_abnormal_idx['indices'], device=device)]
    selected_indices_NN   = [torch.tensor(test_abnormal_idx['indices'], device=device)]

    # 데이터 텐서 변환
    x_dev_normal_list = [torch.tensor(W_dev[idx.cpu().numpy()], dtype=torch.float32).to(device)
                         for idx in selected_indices]
    y_dev_normal_list = [torch.tensor(X_s_dev[idx.cpu().numpy()], dtype=torch.float32).to(device)
                         for idx in selected_indices]

    x_dev_abnormal_list = [torch.tensor(W_dev[idx.cpu().numpy()], dtype=torch.float32).to(device)
                           for idx in selected_indices_test]
    y_dev_abnormal_list = [torch.tensor(X_s_dev[idx.cpu().numpy()], dtype=torch.float32).to(device)
                           for idx in selected_indices_test]

    x_test_abnormal_list = [torch.tensor(W_test[idx.cpu().numpy()], dtype=torch.float32).to(device)
                            for idx in selected_indices_NN]
    y_test_abnormal_list = [torch.tensor(X_s_test[idx.cpu().numpy()], dtype=torch.float32).to(device)
                            for idx in selected_indices_NN]

    # dev normal
    x_dev_normal = torch.cat(x_dev_normal_list, dim=0)
    x_dev_mean   = x_dev_normal.mean(dim=0)
    x_dev_var    = x_dev_normal.var(dim=0, unbiased=False)
    x_dev_std    = x_dev_var.sqrt()

    y_dev_normal = torch.cat(y_dev_normal_list, dim=0)
    y_dev_mean   = y_dev_normal.mean(dim=0)
    y_dev_var    = y_dev_normal.var(dim=0, unbiased=False)
    y_dev_std    = y_dev_var.sqrt()

    x_dev_normal_scaled = (x_dev_normal - x_dev_mean) / x_dev_std
    y_dev_normal_scaled = (y_dev_normal - y_dev_mean) / y_dev_std

    # dev abnormal
    x_dev_abnormal        = torch.cat(x_dev_abnormal_list, dim=0)
    y_dev_abnormal        = torch.cat(y_dev_abnormal_list, dim=0)
    x_dev_abnormal_scaled = (x_dev_abnormal - x_dev_mean) / x_dev_std
    y_dev_abnormal_scaled = (y_dev_abnormal - y_dev_mean) / y_dev_std

    # test abnormal
    x_test_abnormal        = torch.cat(x_test_abnormal_list, dim=0)
    y_test_abnormal        = torch.cat(y_test_abnormal_list, dim=0)
    x_test_abnormal_scaled = (x_test_abnormal - x_dev_mean) / x_dev_std
    y_test_abnormal_scaled = (y_test_abnormal - y_dev_mean) / y_dev_std

    return {
        'device': device,
        # 스케일링 통계값
        'x_dev_mean': x_dev_mean, 'x_dev_std': x_dev_std,
        'y_dev_mean': y_dev_mean, 'y_dev_std': y_dev_std,
        # dev normal
        'x_dev_normal': x_dev_normal, 'y_dev_normal': y_dev_normal,
        'x_dev_normal_scaled': x_dev_normal_scaled, 'y_dev_normal_scaled': y_dev_normal_scaled,
        # dev abnormal
        'x_dev_abnormal': x_dev_abnormal, 'y_dev_abnormal': y_dev_abnormal,
        'x_dev_abnormal_scaled': x_dev_abnormal_scaled, 'y_dev_abnormal_scaled': y_dev_abnormal_scaled,
        # test abnormal
        'x_test_abnormal': x_test_abnormal, 'y_test_abnormal': y_test_abnormal,
        'x_test_abnormal_scaled': x_test_abnormal_scaled, 'y_test_abnormal_scaled': y_test_abnormal_scaled,
    }