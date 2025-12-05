#!/usr/bin/env python3
import os
import sys
import re

# Import our worker modules
import analyzer
import comparator

# === CONFIG ===
# You can change this or pass it as an argument
DEFAULT_ROOT = "F:/particle-data"

# === STRICT FILTERS ===
ALLOWED_RPMS = ["400", "700", "1000"]
ALLOWED_DURS = ["12mins", "60mins"]

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = DEFAULT_ROOT

    if not os.path.isdir(root_dir):
        print(f"❌ Root directory not found: {root_dir}")
        sys.exit(1)

    print(f"🚀 STARTING STRICT BATCH PROCESSING ON: {root_dir}")
    print("------------------------------------------------")

    # 1. SCAN LEVEL 1: MATERIALS
    # We assume any folder in the root is a "Material"
    material_folders = [f for f in os.scandir(root_dir) if f.is_dir()]

    for mat in material_folders:
        print(f"\n📦 Material: {mat.name}")
        
        # 2. SCAN LEVEL 2: RPM GROUPS (400, 700, 1000)
        rpm_folders = [f for f in os.scandir(mat.path) if f.is_dir() and f.name in ALLOWED_RPMS]
        
        if not rpm_folders:
            continue

        for rpm in rpm_folders:
            print(f"   ⚙️ RPM Group: {rpm.name}")
            
            # 3. SCAN LEVEL 3: DURATIONS (12mins, 60mins)
            dur_folders = [f for f in os.scandir(rpm.path) if f.is_dir() and f.name in ALLOWED_DURS]
            
            processed_durations = False
            
            for dur in dur_folders:
                # 4. CHECK LEVEL 4: TRIALS (-T#)
                # We perform a quick check to see if valid trials exist before running analysis
                # Regex looks for "Ends with -T followed by digits"
                trials = [t for t in os.scandir(dur.path) if t.is_dir() and re.search(r'-T\d+$', t.name)]
                
                if trials:
                    print(f"      ⏱️ Analyzing {dur.name} ({len(trials)} trials found)...")
                    analyzer.run_analysis(dur.path)
                    processed_durations = True
                else:
                    print(f"      ⚠️ Skipping {dur.name} (No valid '-T#' folders found)")

            # IF we successfully processed data in this RPM group, run the comparison
            if processed_durations:
                print(f"      📊 Comparing Runs for {mat.name} @ {rpm.name} RPM...")
                comparator.run_comparison(rpm.path)

    print("\n✨ BATCH RUN COMPLETE ✨")

if __name__ == "__main__":
    main()