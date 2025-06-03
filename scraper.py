import requests
import PyPDF2
import io
import pandas as pd
from typing import List
import sys
import re

def download_pdf(url: str) -> bytes:
    """Download PDF from URL."""
    print(f"Attempting to download PDF from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    print(f"Successfully downloaded PDF, size: {len(response.content)} bytes")
    return response.content

def extract_codes_from_pdf(pdf_content: bytes) -> List[str]:
    """Extract Código BYMA column from PDF content."""
    codes = []
    # Pattern to match 1-4 uppercase letters before a market name
    market_names = set([
        'NYSE', 'NASDAQ', 'XETRA', 'BOVESPA', 'AMEX', 'CBOE', 'CME', 'LSE', 'TSX', 'BMV', 'EURONEXT', 'BATS', 'GS', 'ARCA', 'BME', 'FWB', 'SIX', 'TSE', 'HKEX', 'SGX', 'KRX', 'ASX', 'JSE', 'BSE', 'NSE', 'MOEX', 'SZSE', 'SSE', 'TWSE', 'BM&FBOVESPA', 'B3', 'BVL', 'BCBA', 'BYMA', 'MERVAL', 'BCS', 'BCBA', 'BCP', 'BVC', 'BVN', 'BVMB', 'BVPA', 'BVSA', 'BVSP', 'BVX', 'BVZ', 'BZX', 'BMF', 'BMFBOVESPA', 'BMV', 'BOV', 'BOVESPA', 'BSE', 'BVC', 'BVL', 'BVMF', 'BVP', 'BVQ', 'BVX', 'BVZ', 'CBOE', 'CME', 'EUREX', 'EURONEXT', 'FWB', 'HKEX', 'ICE', 'JSE', 'KRX', 'LSE', 'MOEX', 'NASDAQ', 'NYSE', 'SGX', 'SIX', 'SSE', 'SZSE', 'TSE', 'TWSE', 'XETRA'
    ])
    ratio_pattern = re.compile(r'\d+:\d+')
    pdf_file = io.BytesIO(pdf_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    print(f"PDF has {len(pdf_reader.pages)} pages")
    debug_lines = []
    debug_candidates = []
    for page_num, page in enumerate(pdf_reader.pages, 1):
        print(f"Processing page {page_num}")
        text = page.extract_text()
        if not text:
            print(f"Warning: No text extracted from page {page_num}")
            continue
        lines = text.split('\n')
        print(f"Found {len(lines)} lines on page {page_num}")
        for line_num, line in enumerate(lines, 1):
            if len(debug_lines) < 20:
                debug_lines.append(line)
            # Skip header or separator lines
            if ("Código" in line and "BYMA" in line) or ("---" in line) or ("Nombre" in line) or ("Cotiza Ratio" in line):
                continue
            # Tokenize the line
            tokens = line.split()
            for i in range(1, len(tokens)):
                token = tokens[i]
                prev_token = tokens[i-1]
                if token in market_names:
                    # Check if previous token is a valid ticker
                    if re.fullmatch(r'[A-Z]{1,4}', prev_token):
                        if not ratio_pattern.match(prev_token):
                            if len(debug_candidates) < 20:
                                debug_candidates.append(prev_token)
                            codes.append(prev_token)
                            if len(codes) % 10 == 0:
                                print(f"Found {len(codes)} codes so far...")
    print("First 20 lines from PDF text:")
    for l in debug_lines:
        print(repr(l))
    print("First 20 code candidates:")
    for c in debug_candidates:
        print(repr(c))
    unique_codes = sorted(list(set(codes)))
    print(f"Total codes found: {len(codes)}, unique codes: {len(unique_codes)}")
    return unique_codes

def main():
    # PDF URL
    url = "https://cdn.prod.website-files.com/6697a441a50c6b926e1972e0/682dec4f842af0ab2d2034f5_BYMA-Tabla-CEDEARs-2025-05-15.pdf"
    
    try:
        # Download PDF
        print("Downloading PDF...")
        pdf_content = download_pdf(url)
        
        # Extract codes
        print("Extracting codes...")
        codes = extract_codes_from_pdf(pdf_content)
        
        if not codes:
            print("Warning: No codes were extracted from the PDF")
            return
            
        # Create DataFrame and save to CSV
        df = pd.DataFrame(codes, columns=['Código BYMA'])
        output_file = 'byma_codes.csv'
        df.to_csv(output_file, index=False, header=False)
        
        print(f"Successfully extracted {len(codes)} codes to {output_file}")
        print("First few codes:", codes[:5])
        print("\nAll extracted codes:")
        for code in codes:
            print(code)
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {str(e)}")
    except PyPDF2.PdfReadError as e:
        print(f"Error reading PDF: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
