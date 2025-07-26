#!/usr/bin/env python3
import json


def compare_notebooks(file1, file2):
    with open(file1) as f1:
        nb1 = json.load(f1)
    with open(file2) as f2:
        nb2 = json.load(f2)

    # Check metadata differences
    if nb1.get("metadata") != nb2.get("metadata"):
        print("Metadata differs")
        print("First:", nb1.get("metadata"))
        print("Second:", nb2.get("metadata"))

    # Check cell differences
    for i, (c1, c2) in enumerate(zip(nb1["cells"], nb2["cells"], strict=False)):
        if c1 != c2:
            print(f"\nCell {i} differs")
            # Check source
            if c1.get("source") != c2.get("source"):
                print("Source differs")
            # Check metadata
            if c1.get("metadata") != c2.get("metadata"):
                print("Metadata differs:", c1.get("metadata"), "vs", c2.get("metadata"))
            # Check execution count
            if c1.get("execution_count") != c2.get("execution_count"):
                print("Execution count differs:", c1.get("execution_count"), "vs", c2.get("execution_count"))
            # Check ID
            if c1.get("id") != c2.get("id"):
                print("ID differs:", c1.get("id"), "vs", c2.get("id"))


if __name__ == "__main__":
    file1 = "/tmp/test1/simple_sales/Monthly_Sales.ipynb"
    file2 = "/tmp/test2/simple_sales/Monthly_Sales.ipynb"
    compare_notebooks(file1, file2)
