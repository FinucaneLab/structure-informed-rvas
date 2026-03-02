#!/usr/bin/env python

import argparse
import csv
from pymol import cmd

def main():
    parser = argparse.ArgumentParser(
        description="Color a specific chain in an mmCIF structure by per-residue ratio values"
    )
    parser.add_argument("--tsv", required=True, help="TSV file with uniprot_id, aa_pos, ratio columns")
    parser.add_argument("--input-cif", required=True, help="Input structure (.cif)")
    parser.add_argument("--output-pse", required=True, help="Output PyMOL session (.pse)")
    parser.add_argument("--chain", required=True, help="Target chain ID (e.g. A)")
    parser.add_argument("--uniprot", required=True, help="Target UniProt ID")

    args = parser.parse_args()

    # Reset PyMOL
    cmd.reinitialize()

    # Load mmCIF structure
    cmd.load(args.input_cif, "complex")

    # Cartoon-only display
    cmd.hide("everything", "all")
    cmd.show("cartoon", "all")

    # Grey + transparent background chains
    cmd.color("grey70", "all")
    cmd.set("cartoon_transparency", 0.6, "all")
    cmd.bg_color("white")

    # Read ratio TSV
    residue_ratios = {}

    with open(args.tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["uniprot_id"] == args.uniprot:
                try:
                    aa_pos = int(row["aa_pos"])
                    ratio = float(row["ratio"])
                    residue_ratios[aa_pos] = ratio
                except ValueError:
                    continue

    if not residue_ratios:
        raise RuntimeError(f"No ratio data found for UniProt ID {args.uniprot}")

    # Normalize ratios EXACTLY like pymol_scan_test
    max_ratio = max(residue_ratios.values())
    if max_ratio == 0:
        max_ratio = 1.0

    # Write normalized ratios into B-factors
    for resi, ratio in residue_ratios.items():
        ratio_norm = ratio / max_ratio
        selection = f"chain {args.chain} and resi {resi}"
        cmd.alter(selection, f"b={ratio_norm}")

    cmd.rebuild()

    # Apply identical color spectrum
    cmd.spectrum(
        "b",
        "green_red",
        f"chain {args.chain}",
        byres=1
    )

    # Make target chain opaque
    cmd.set("cartoon_transparency", 0.0, f"chain {args.chain}")

    # Save PyMOL session
    cmd.save(args.output_pse)

    print(f"[OK] Saved colored session to {args.output_pse}")

if __name__ == "__main__":
    main()
