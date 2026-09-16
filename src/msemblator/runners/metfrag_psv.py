"""Adapt CSV-style PSV input to MetFrag's physical-line, pipe-split reader."""

import csv
import re


def validate_row(row, headers, path, line_number):
    location = f"{path}: line {line_number}"
    if len(row) != len(headers):
        raise ValueError(
            f"{location}: expected {len(headers)} PSV columns, got {len(row)}"
        )
    result = []
    for header, cell in zip(headers, row):
        if "\r" in cell or "\n" in cell:
            if header not in {"Identifier", "CompoundName", "DTXSID", "MAPPED_DTXSID",
                              "PREFERRED_NAME_DTXSID"}:
                raise ValueError(f"{location}: newline in {header}")
            cell = re.sub(r"[\r\n]+", " ", cell)
        if "|" in cell:
            raise ValueError(f"{location}: pipe inside {header}; MetFrag cannot read quoted pipes")
        result.append(cell)
    # Java String.split drops trailing empty fields.
    if not result[-1]:
        raise ValueError(f"{location}: empty final PSV column ({headers[-1]})")
    return result


def read_psv(path):
    """Yield (physical line number, row), skipping blank rows and removing BOM."""
    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file, delimiter="|", strict=True)
        try:
            for row in reader:
                if not row or all(not cell.strip() for cell in row):
                    continue
                yield reader.line_num, row
        except csv.Error as exc:
            raise ValueError(f"{path}: line {reader.line_num}: invalid PSV: {exc}") from exc


def read_header(rows, path):
    _, headers = next(rows, (0, []))
    if len(headers) < 2 or len(set(headers)) != len(headers):
        raise ValueError(f"{path}: missing or invalid PSV header")
    return headers


def write_psv(path, headers, rows):
    """Write LF endings with no embedded line breaks or pipes in any field."""
    # Validate before opening the destination so invalid input is not truncated.
    validated = [validate_row(row, headers, path, i) for i, row in enumerate(rows, 2)]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="|", lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(validated)
