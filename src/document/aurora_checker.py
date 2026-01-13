"""
Utility functions for checking AURORA's SEC filing data to avoid duplicates.

This module provides functions that NOVA can use to check if SEC filings
already exist in AURORA's data directory before downloading.
"""
from pathlib import Path
from typing import Set, Tuple, Optional


def get_aurora_data_dir() -> Path:
    """
    Get the AURORA data directory path.
    
    Returns:
        Path to AURORA's data directory
    """
    # AURORA data directory is at /Users/bk/AURORA/AURORA/data
    return Path("/Users/bk/AURORA/AURORA/data")


def check_filing_exists_in_aurora(
    cik: str,
    form_type: str,
    filing_date: str,
    accession_number: str,
    aurora_data_dir: Optional[Path] = None
) -> bool:
    """
    Check if a specific SEC filing already exists in AURORA's data directory.
    
    This function can be used by NOVA to avoid downloading duplicate filings.
    
    Args:
        cik: Company CIK number (with or without leading zeros)
        form_type: Form type (e.g., "10-K", "10-Q", "8-K")
        filing_date: Filing date in YYYY-MM-DD format
        accession_number: SEC accession number (e.g., "0001628280-25-055234")
        aurora_data_dir: Optional path to AURORA data directory. If None, uses default.
        
    Returns:
        True if filing exists in AURORA, False otherwise
        
    Example:
        >>> check_filing_exists_in_aurora(
        ...     cik="1784570",
        ...     form_type="10-Q",
        ...     filing_date="2025-12-04",
        ...     accession_number="0001628280-25-055234"
        ... )
        True
    """
    if aurora_data_dir is None:
        aurora_data_dir = get_aurora_data_dir()
    
    raw_dir = aurora_data_dir / "raw"
    cik_clean = cik.lstrip("0")
    company_dir = raw_dir / cik_clean
    
    if not company_dir.exists():
        return False
    
    # Construct expected filename (AURORA format: FORM_DATE_ACCESSION.html)
    form_clean = form_type.replace(" ", "_")
    expected_filename = f"{form_clean}_{filing_date}_{accession_number}.html"
    expected_path = company_dir / expected_filename
    
    return expected_path.exists()


def get_existing_filings_set(
    cik: str,
    aurora_data_dir: Optional[Path] = None
) -> Set[Tuple[str, str, str]]:
    """
    Get a set of existing filing identifiers for quick lookup.
    
    Returns a set of tuples: (form_type, filing_date, accession_number)
    This is useful for NOVA to quickly check if a filing exists.
    
    Args:
        cik: Company CIK number
        aurora_data_dir: Optional path to AURORA data directory. If None, uses default.
        
    Returns:
        Set of tuples (form_type, filing_date, accession_number)
        
    Example:
        >>> existing = get_existing_filings_set("1784570")
        >>> ("10-Q", "2025-12-04", "0001628280-25-055234") in existing
        True
    """
    if aurora_data_dir is None:
        aurora_data_dir = get_aurora_data_dir()
    
    raw_dir = aurora_data_dir / "raw"
    cik_clean = cik.lstrip("0")
    company_dir = raw_dir / cik_clean
    
    if not company_dir.exists():
        return set()
    
    existing_filings = set()
    
    for file_path in company_dir.glob("*.html"):
        # Parse filename: FORM_DATE_ACCESSION.html
        parts = file_path.stem.split("_")
        if len(parts) >= 3:
            form_type = parts[0].replace("_", " ")
            filing_date = parts[1]
            accession = "_".join(parts[2:])
            
            existing_filings.add((form_type, filing_date, accession))
    
    return existing_filings


def filter_existing_filings(
    cik: str,
    filings_to_check: list,
    aurora_data_dir: Optional[Path] = None
) -> list:
    """
    Filter out filings that already exist in AURORA.
    
    This function takes a list of filings (with form_type, filing_date, accession_number)
    and returns only those that don't exist in AURORA.
    
    Args:
        cik: Company CIK number
        filings_to_check: List of dicts or tuples with filing info.
                         Each item should have: form_type, filing_date, accession_number
        aurora_data_dir: Optional path to AURORA data directory. If None, uses default.
        
    Returns:
        List of filings that don't exist in AURORA
        
    Example:
        >>> filings = [
        ...     {"form_type": "10-Q", "filing_date": "2025-12-04", "accession_number": "0001628280-25-055234"},
        ...     {"form_type": "10-K", "filing_date": "2024-12-31", "accession_number": "0001234567-24-000001"}
        ... ]
        >>> new_filings = filter_existing_filings("1784570", filings)
    """
    existing = get_existing_filings_set(cik, aurora_data_dir)
    
    new_filings = []
    for filing in filings_to_check:
        if isinstance(filing, dict):
            key = (filing["form_type"], filing["filing_date"], filing["accession_number"])
        else:
            # Assume tuple
            key = tuple(filing[:3])
        
        if key not in existing:
            new_filings.append(filing)
    
    return new_filings
