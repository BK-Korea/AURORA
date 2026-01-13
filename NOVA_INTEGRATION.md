# NOVA와 AURORA 통합 가이드

NOVA가 AURORA의 SEC filing 데이터를 확인하고 중복 다운로드를 방지하는 방법입니다.

## 개요

AURORA는 `/Users/bk/AURORA/AURORA/data/raw/{CIK}/` 경로에 SEC filing을 저장합니다.
파일명 형식: `{form_type}_{filing_date}_{accession_number}.html`

예: `10-Q_2025-12-04_0001628280-25-055234.html`

## 사용 방법

### 방법 1: `aurora_checker` 모듈 사용 (권장)

```python
from src.document.aurora_checker import (
    check_filing_exists_in_aurora,
    get_existing_filings_set,
    filter_existing_filings
)

# 개별 파일 확인
exists = check_filing_exists_in_aurora(
    cik="1784570",
    form_type="10-Q",
    filing_date="2025-12-04",
    accession_number="0001628280-25-055234"
)

if not exists:
    # 다운로드 진행
    download_filing(...)

# 기존 파일 목록 가져오기 (빠른 조회용)
existing = get_existing_filings_set("1784570")
# Returns: {("10-Q", "2025-12-04", "0001628280-25-055234"), ...}

# 다운로드할 파일 목록 필터링
filings_to_download = [
    {"form_type": "10-Q", "filing_date": "2025-12-04", "accession_number": "0001628280-25-055234"},
    {"form_type": "10-K", "filing_date": "2024-12-31", "accession_number": "0001234567-24-000001"}
]

new_filings = filter_existing_filings("1784570", filings_to_download)
# 이미 존재하는 파일은 제외됨
```

### 방법 2: `SECDownloader` 클래스 사용

```python
from src.document.sec_downloader import SECDownloader
from pathlib import Path

# AURORA 데이터 디렉토리 경로
aurora_data_dir = Path("/Users/bk/AURORA/AURORA/data")
downloader = SECDownloader(data_dir=aurora_data_dir)

# 개별 파일 확인
exists = downloader.check_filing_exists(
    cik="1784570",
    form_type="10-Q",
    filing_date="2025-12-04",
    accession_number="0001628280-25-055234"
)

# 기존 파일 목록 가져오기
existing = downloader.get_existing_filings_set("1784570")
```

## NOVA 통합 예시

```python
# NOVA의 SEC 다운로더에서 사용
def download_sec_filings_nova(cik: str, form_types: list):
    from src.document.aurora_checker import (
        get_existing_filings_set,
        check_filing_exists_in_aurora
    )
    
    # 기존 파일 목록 가져오기
    existing = get_existing_filings_set(cik)
    
    # SEC에서 파일 목록 가져오기
    sec_filings = get_filings_from_sec(cik, form_types)
    
    # 중복 제거
    new_filings = []
    for filing in sec_filings:
        key = (filing.form_type, filing.filing_date, filing.accession_number)
        if key not in existing:
            new_filings.append(filing)
    
    # 새 파일만 다운로드
    for filing in new_filings:
        download_filing(filing)
```

## 파일 구조

AURORA의 SEC filing 저장 구조:
```
/Users/bk/AURORA/AURORA/data/raw/
├── 1784570/                    # CIK별 폴더
│   ├── 10-Q_2025-12-04_0001628280-25-055234.html
│   ├── 8-K_2025-11-06_0001193125-25-269718.html
│   └── 8-K_2025-12-04_0001628280-25-055206.html
└── {other_cik}/
    └── ...
```

## 주의사항

1. **CIK 형식**: AURORA는 앞의 0을 제거한 CIK를 사용합니다 (예: "1784570").
2. **파일명 형식**: `{form_type}_{filing_date}_{accession_number}.html`
   - form_type: 공백은 언더스코어로 변환 (예: "10-Q", "DEF 14A" → "DEF_14A")
   - filing_date: YYYY-MM-DD 형식
   - accession_number: SEC accession number (예: "0001628280-25-055234")
3. **경로**: 기본 경로는 `/Users/bk/AURORA/AURORA/data/raw/`입니다.

## 테스트

```python
# 테스트 예시
from src.document.aurora_checker import check_filing_exists_in_aurora

# 실제 존재하는 파일 확인
result = check_filing_exists_in_aurora(
    cik="1784570",
    form_type="10-Q",
    filing_date="2025-12-04",
    accession_number="0001628280-25-055234"
)
print(f"File exists: {result}")  # True
```
