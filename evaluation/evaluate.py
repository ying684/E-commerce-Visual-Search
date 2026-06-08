# File: evaluation/evaluate.py
import torch
import faiss
import numpy as np
from tqdm import tqdm

def calculate_map_at_k(predictions, targets, k=5):
    """Tính toán mAP@K cho một batch các truy vấn"""
    map_sum = 0.0
    for pred, target in zip(predictions, targets):
        # pred: list of posting_ids model dự đoán
        # target: list of posting_ids thực sự giống nhau
        target_set = set(target)
        hits = 0
        sum_precisions = 0
        
        for i, p in enumerate(pred[:k]):
            if p in target_set:
                hits += 1
                sum_precisions += hits / (i + 1.0)
                
        if hits > 0:
            map_sum += sum_precisions / min(len(target_set), k)
            
    return map_sum / len(predictions)

def evaluate_model(model, dataloader, device, k=5):
    """
    Đánh giá model trên tập Validation.
    Tất cả ảnh trong Val vừa làm thư viện (Gallery), vừa làm truy vấn (Query).
    """
    model.eval()
    all_embeddings = []
    all_posting_ids = []
    all_label_groups = []
    
    print("\n[*] Đang trích xuất vector tập Validation để đánh giá...")
    with torch.no_grad():
        # ĐÃ SỬA Ở ĐÂY: Thêm 'labels' vào để hứng đủ 4 giá trị từ ArcFace Dataset
        for images, labels, posting_ids, label_groups in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)
            embeddings = model(images).cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_posting_ids.extend(posting_ids)
            all_label_groups.extend(label_groups.numpy())
            
    all_embeddings = np.vstack(all_embeddings).astype('float32')
    
    # Xây dựng index FAISS (Gallery)
    index = faiss.IndexFlatL2(all_embeddings.shape[1])
    try:
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
    except AttributeError:
        # Tự động Fallback: Nếu thư viện là faiss-cpu, hệ thống sẽ bỏ qua và chạy ngầm bằng CPU
        pass 
        
    index.add(all_embeddings)
    
    # Tìm kiếm (Query)
    print(f"[*] Đang truy vấn FAISS để tính mAP@{k}...")
    _, indices = index.search(all_embeddings, k + 1) # Lấy k+1 vì kết quả đầu tiên luôn là chính nó
    
    # Tính điểm mAP
    predictions = []
    targets = []
    
    # Tạo từ điển nhóm sản phẩm để biết ground truth
    group_dict = {}
    for i, group in enumerate(all_label_groups):
        if group not in group_dict:
            group_dict[group] = []
        group_dict[group].append(all_posting_ids[i])
        
    for i in range(len(all_posting_ids)):
        # Bỏ qua kết quả đầu tiên (chính nó)
        pred_ids = [all_posting_ids[idx] for idx in indices[i][1:k+1]]
        target_ids = group_dict[all_label_groups[i]]
        # Bỏ id của chính nó ra khỏi target
        target_ids = [tid for tid in target_ids if tid != all_posting_ids[i]]
        
        predictions.append(pred_ids)
        targets.append(target_ids)
        
    map_score = calculate_map_at_k(predictions, targets, k)
    return map_score