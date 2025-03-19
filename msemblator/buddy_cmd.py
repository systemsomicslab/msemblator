import os
import pandas as pd
from msbuddy import Msbuddy, MsbuddyConfig, Adduct


def run_msbuddy(input_dir, output_dir, batch_size=1000):
    """
    Run MSBuddy to annotate molecular formulas from MGF files and save the results in chunks of specified size.
    """
    # Create an MsbuddyConfig object with specific configuration parameters
    msb_config = MsbuddyConfig(
        ms_instr=None,
        ppm=True,
        ms1_tol=10,
        ms2_tol=20,
        timeout_secs=600,
        halogen=True
    )

    # Initialize the Msbuddy engine with the specified configuration
    msb_engine = Msbuddy(msb_config)
    print("Updated MS1 tolerance:", msb_engine.config.ms1_tol)
    print("Updated MS2 tolerance:", msb_engine.config.ms2_tol)

    # Retrieve all .mgf files from the input directory
    mgf_files = [f for f in os.listdir(input_dir) if f.endswith('.mgf')]

    for mgf_file in mgf_files:
        # Extract the adduct name from the file name (e.g., [M-H]-)
        adduct_name = os.path.splitext(mgf_file)[0]  
        
        # Determine the ion mode: '+' indicates positive mode, '-' indicates negative mode
        if adduct_name.endswith('+'):
            pos_ion_mode = True
        elif adduct_name.endswith('-'):
            pos_ion_mode = False
        else:
            print(f"Invalid adduct sign in file: {mgf_file}")
            continue
        
        # Load the MGF file into the Msbuddy engine
        mgf_file_path = os.path.join(input_dir, mgf_file)
        msb_engine.load_mgf(mgf_file_path)

        # Assign the adduct type to each feature in the data
        for feature in msb_engine.data:
            feature.adduct = Adduct(adduct_name, pos_mode=pos_ion_mode)

        # Split data into chunks of batch_size
        data_chunks = [msb_engine.data[i:i + batch_size] for i in range(0, len(msb_engine.data), batch_size)]
        
        for idx, chunk in enumerate(data_chunks, start=1):
            # Assign chunk to engine and process
            msb_engine.data = chunk
            msb_engine.annotate_formula()

            # Retrieve summary results for the chunk
            summary_df = pd.DataFrame(msb_engine.get_summary())

            # Identify correct m/z column
            mz_column = next((col for col in ["m/z", "mz", "precursor_mz"] if col in summary_df.columns), None)
            if mz_column is None:
                raise KeyError("Could not find a valid m/z column in summary_df.")
            if "identifier" not in summary_df.columns:
                raise KeyError("The column 'identifier' is missing from summary_df.")
            
            # Create identifier mapping
            identifier_map = dict(zip(summary_df[mz_column], summary_df["identifier"]))

            # Retrieve detailed candidate results
            detailed_result_list = []
            for meta_feature in chunk:
                scan_id = identifier_map.get(meta_feature.mz, None)
                rt_value = getattr(meta_feature, "rt", None)

                # Ensure candidate_formula_list is not None
                candidate_list = getattr(meta_feature, "candidate_formula_list", [])
                if not isinstance(candidate_list, list):
                    print(f"Warning: candidate_formula_list is not a list for m/z {meta_feature.mz}. Skipping this feature.")
                    continue

                for i, candidate in enumerate(candidate_list):
                    detailed_result_list.append({
                        "Scan_ID": scan_id,
                        "m/z": meta_feature.mz,
                        "RT": rt_value,
                        "Rank": i+1,
                        "Formula": candidate.formula,
                        "MLR_score": getattr(candidate, "mlr_score", None),
                        "Estimated_FDR": getattr(candidate, "estimated_fdr", None)
                    })

            # Save the chunk's summary and detailed results
            summary_csv = os.path.join(output_dir, f'summary_{adduct_name}_batch{idx}.csv')
            detailed_csv = os.path.join(output_dir, f'detailed_summary_{adduct_name}_batch{idx}.csv')
            
            summary_df.to_csv(summary_csv, index=False)
            pd.DataFrame(detailed_result_list).to_csv(detailed_csv, index=False)

            print(f"Processed and saved batch {idx} for {mgf_file_path} as {summary_csv} and {detailed_csv}")