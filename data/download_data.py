"""
data/download_data.py
Tự động tải bộ dữ liệu UIT-VSFC chính thức từ Hugging Face về thư mục data/uit_vsfc/.
"""

import os
import sys
sys.dont_write_bytecode = True
import urllib.request
from typing import Dict, List, Optional

HF_PARQUET_URLS = {
    "train": "https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default/train/0.parquet",
    "validation": "https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default/validation/0.parquet",
    "test": "https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default/test/0.parquet",
}

EXPECTED_ROW_COUNTS = {
    "train": 11426,
    "validation": 1583,
    "test": 3166,
}


def download_vsfc_dataset(output_dir: str = "data/uit_vsfc", force: bool = False) -> Dict[str, str]:
    """Tải và lưu trữ 3 split Parquet của UIT-VSFC."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    print("=" * 65 + f"\nTẢI BỘ DỮ LIỆU UIT-VSFC TIẾNG VIỆT -> '{output_dir}'\n" + "=" * 65)

    headers = {"User-Agent": "Mozilla/5.0"}
    for split, url in HF_PARQUET_URLS.items():
        dst_path = os.path.join(output_dir, f"{split}.parquet")
        if os.path.isfile(dst_path) and os.path.getsize(dst_path) > 1000 and not force:
            print(f"[*] {split:10s}: Đã có sẵn ({os.path.getsize(dst_path):,} bytes) -> '{dst_path}'")
            results[split] = dst_path
            continue

        print(f"[+] {split:10s}: Đang tải từ Hugging Face...")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp, open(dst_path, "wb") as f:
            f.write(resp.read())
        print(f"[OK] {split:10s}: Đã tải xong ({os.path.getsize(dst_path):,} bytes) -> '{dst_path}'")
        results[split] = dst_path

    # Kiểm tra số dòng bằng pandas
    try:
        import pandas as pd
        for split, path in results.items():
            count = len(pd.read_parquet(path))
            exp = EXPECTED_ROW_COUNTS.get(split, 0)
            status = "MATCH" if count == exp else f"MISMATCH (expected {exp})"
            print(f"    Xác thực {split:10s}: {count:,} dòng [{status}]")
    except Exception:
        pass

    print("=" * 65 + "\nTải dữ liệu hoàn tất.\n")
    return results


if __name__ == "__main__":
    download_vsfc_dataset()
